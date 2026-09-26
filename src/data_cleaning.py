import pandas as pd
import numpy as np

NON_PRODUCT_CHARGE_CODES = {'POST', 'DOT', 'C2', 'D', 'BANK CHARGES', 'M'}
TEST_CODES = {'TEST001', 'TEST002'}
BAD_DEBT_CODE = {'B'}
STAFF_ADJUST_CODE = {'ADJUST'}
MARKETPLACE_FEE_CODE = {'AMAZONFEE'}
SAMPLE_CODE = {'S'}

def load_raw(path: str) -> pd.DataFrame:
    """Load both sheets of the raw UCI Online Retail II file and concatenate them,
    tagging origin so the sheet-overlap issue (Dec 1-9, 2010) can be diagnosed."""
    sheet1 = pd.read_excel(path, sheet_name='Year 2009-2010')
    sheet2 = pd.read_excel(path, sheet_name='Year 2010-2011')
    sheet1['SourceSheet'] = 'Year 2009-2010'
    sheet2['SourceSheet'] = 'Year 2010-2011'
    return pd.concat([sheet1, sheet2], ignore_index=True)

def clean_transactions(raw: pd.DataFrame) -> dict:
    """
    Apply the finalized Stage 2 cleaning rules and return a dict of three
    analytical views: sales_view, rfm_base, product_view. Also returns
    transactions_core (the shared base layer) and an audit trail list.

    Rules applied, in order (see Stage 2 report for full rationale):
      1+2. Global drop_duplicates() on business columns resolves both the
           Dec 1-9 2010 sheet overlap and ordinary duplicate rows in one step.
      3.   Cancellations (Invoice starts with 'C') are KEPT and flagged, not removed.
      4/6. Internal write-offs = Price==0 AND Customer ID missing -> excluded.
      5.   Bad-debt entries (StockCode='B') -> excluded.
      8.   Non-product charge codes (POST/DOT/C2/D/BANK CHARGES/M) -> excluded
           only from the product view, kept in the sales view (real money flow).
      9.   StockCode case normalized into StockCode_clean for grouping only.
      10.  Non-country labels ('Unspecified','European Community') mapped to
           'Unknown/Other' in Country_clean; raw Country left untouched.
      12.  Test transactions (TEST001/TEST002) -> excluded (found during
           validation, not in the original Stage 1 issue list).
    """
    audit = []
    business_cols = [c for c in raw.columns if c != 'SourceSheet']

    df = raw.drop_duplicates(subset=business_cols, keep='first').copy()
    audit.append(('Duplicate/overlap rows removed', len(raw) - len(df), len(df)))

    df['Invoice_str'] = df['Invoice'].astype(str)
    df['IsCancellation'] = df['Invoice_str'].str.startswith('C')
    df['StockCode_clean'] = df['StockCode'].astype(str).str.upper().str.strip()
    df['Country_clean'] = df['Country'].replace({
        'Unspecified': 'Unknown/Other',
        'European Community': 'Unknown/Other'
    })
    df['LineTotal'] = df['Quantity'] * df['Price']

    for name, codes in [
        ('Test transactions', TEST_CODES),
        ('Bad-debt entries', BAD_DEBT_CODE),
        ('Staff adjustment entries', STAFF_ADJUST_CODE),
        ('Marketplace fee entries', MARKETPLACE_FEE_CODE),
        ('Internal sample stock', SAMPLE_CODE),
    ]:
        mask = df['StockCode_clean'].isin(codes)
        audit.append((f'{name} removed', int(mask.sum()), None))
        df = df[~mask].copy()

    mask_writeoff = (df['Price'] == 0) & (df['Customer ID'].isna())
    audit.append(('Internal write-offs (Price=0, no Customer ID) removed', int(mask_writeoff.sum()), len(df) - int(mask_writeoff.sum())))
    df = df[~mask_writeoff].copy()
    df['IsFreeItem'] = (df['Price'] == 0) & (df['Customer ID'].notna())

    transactions_core = df.copy()

    sales_view = transactions_core.copy()

    mask_no_cust = sales_view['Customer ID'].isna()
    rfm_base = sales_view[~mask_no_cust].copy()

    mask_nonproduct = sales_view['StockCode_clean'].isin(NON_PRODUCT_CHARGE_CODES)
    product_view = sales_view[~mask_nonproduct].copy()

    return {
        'transactions_core': transactions_core,
        'sales_view': sales_view,
        'rfm_base': rfm_base,
        'product_view': product_view,
        'audit': audit,
    }
