from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
from pydantic import Field, BaseModel
from datetime import date
from tqdm import tqdm
from tenacity import retry
from json import loads
from pandas import DataFrame
from enum import Enum
from typing import List

BASE_DATA_PATH = "./data/"
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class District(str, Enum):
    Colombo = "Colombo"
    Gampaha = "Gampaha"
    Kalutara = "Kalutara"
    Kandy = "Kandy"
    Matale = "Matale"
    Nuwara_Eliya = "Nuwara Eliya"
    Galle = "Galle"
    Hambantota = "Hambantota"
    Matara = "Matara"
    Jaffna = "Jaffna"
    Kilinochchi = "Kilinochchi"
    Mannar = "Mannar"
    Vavuniya = "Vavuniya"
    Mullaitivu = "Mullaitivu"
    Batticaloa = "Batticaloa"
    Ampara = "Ampara"
    Trincomalee = "Trincomalee"
    Kalmunai = "Kalmunai"
    Kurunegala = "Kurunegala"
    Puttalam = "Puttalam"
    Anuradhapura = "Anuradhapura"
    Polonnaruwa = "Polonnaruwa"
    Badulla = "Badulla"
    Moneragala = "Moneragala"
    Ratnapura = "Ratnapura"
    Kegalle = "Kegalle"


class DistrictCases(BaseModel):
    district: District = Field(
        description="RDHS division name as one of the allowed enum values."
    )
    cases: int = Field(
        ge=0,
        description="Column A (current week) Dengue Fever cases for this district.",
    )


class ExtractionData(BaseModel):
    start_date: date = Field(
        description="Week start date from the header (ISO YYYY-MM-DD)."
    )
    end_date: date = Field(
        description="Week end date from the header (ISO YYYY-MM-DD)."
    )
    national_total: int = Field(
        ge=0,
        description="Dengue Fever national total for the CURRENT WEEK (SRI LANKA row, column A).",
    )
    districts: List[DistrictCases] = Field(
        description="Per-RDHS Dengue Fever counts, column A (current week) for Table 1."
    )


prompt = """
ROLE
You are an extractor for Sri Lanka’s Weekly Epidemiological Report (WER) PDF.

HARD FILTERS (read carefully)
- TABLE: Use **Table 1: Selected notifiable diseases** only. Ignore Table 2 or any others.
- DISEASE: Use the **Dengue Fever** row/section only (accept “Dengue” or “DF” synonyms).
- COLUMN: Use **column "A" only** (A = cases during the CURRENT WEEK). **Never** use column "B".
- NATIONAL TOTAL: Prefer the **"SRI LANKA"** total row for Dengue Fever in column A. If that cell is missing/illegible, FALLBACK = sum of column A across all listed districts for Dengue Fever.
- If a value is not from **Dengue Fever → column A**, discard it.

STRICT OUTPUT (schema is enforced by the API; match its names/types exactly)
- start_date (ISO date, YYYY-MM-DD)
- end_date   (ISO date, YYYY-MM-DD)
- national_total (int) = Dengue Fever national total (SRI LANKA row) in **column A**
- districts: list of { district (enum), cases (int) } for these RDHS values only:
  Colombo, Gampaha, Kalutara, Kandy, Matale, Nuwara Eliya, Galle, Hambantota, Matara,
  Jaffna, Kilinochchi, Mannar, Vavuniya, Mullaitivu, Batticaloa, Ampara, Trincomalee,
  Kalmunai, Kurunegala, Puttalam, Anuradhapura, Polonnaruwa, Badulla, Moneragala,
  Ratnapura, Kegalle

WHAT TO READ
1) HEADER (top of first page): week range like “05th – 11th July 2025”.
   - Parse start_date = first day; end_date = last day. Ignore ordinal suffixes (st/nd/rd/th).
   - Convert to ISO YYYY-MM-DD.
   - VALIDATE: end_date - start_date == 6 days (7-day inclusive span). If off-by-one due to typography,
     adjust to the nearest valid 7-day span consistent with the visible range.

2) TABLE 1 → DENGUE FEVER → COLUMN “A” (CURRENT WEEK)
   - Locate the Dengue Fever section. Read **only** the numbers directly under the **“A” column** for:
     each allowed district row and the **SRI LANKA (national total)** row.
   - Orientation quirks: text may be rotated/vertical; “SRI LANKA” may be vertical—still use its **A** cell.
   - Numeric normalization:
       • Remove thin spaces/regular spaces/commas/non-digits.
       • If OCR split digits (e.g., "1  1  6  2"), collapse to "11162".
       • Coerce to non-negative integer; if blank/dash/NA → 0.
   - District list behavior:
       • If a district row is truly missing in Dengue Fever, omit it.
       • Otherwise include it with its column-A integer (0 if blank).
   - NATIONAL TOTAL rule:
       • Use SRI LANKA row, column A. If illegible/missing, compute sum of all extracted district A values.

FINAL VALIDATION BEFORE OUTPUT
- Use **only** Dengue Fever column **A** values for all counts; never touch column B.
- national_total >= 0; all district cases >= 0.
- Return ONLY the JSON object required by the schema (no extra keys, no prose, no markdown).
"""


def get_all_files(path: str = "./data/") -> list[str]:
    return os.listdir(path)


def upload_files(files: list[str]) -> dict[str, types.File]:
    uploads = {}
    for file in tqdm(files):
        path = f"{BASE_DATA_PATH}/{file}"
        uploads[path] = client.files.upload(file=path)
    return uploads


@retry
def process_llm_for_single_file(file: types.File) -> ExtractionData:
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # or "gemini-2.0-flash-lite" if you must
        contents=[file, prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": ExtractionData,  # the model above
            "temperature": 0,
            "max_output_tokens": 1024,  # important for enumerating all districts
        },
    )
    data: ExtractionData = response.parsed
    return data


def process_files(uploaded_files: dict[str, types.File]) -> None:
    data = []
    for _, file in tqdm(uploaded_files.items()):
        extracted_data = process_llm_for_single_file(file)
        data.append(extracted_data.model_dump())
        DataFrame(data).to_csv("./data.csv", index=False)
    return None


if __name__ == "__main__":
    process_files(upload_files(get_all_files()))
