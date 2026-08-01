import pandas as pd
output = pd.read_csv('output.csv')
print(f'Total rows: {len(output)}')
print(f'Columns: {len(output.columns)}')
print(f'claim_status: {output.claim_status.value_counts().to_dict()}')
print(f'issue_type: {output.issue_type.value_counts().to_dict()}')
print(f'object_part: {output.object_part.value_counts().to_dict()}')
print(f'Gemini errors: {output.evidence_standard_met_reason.str.contains("API error").sum()}')
risk_flags=set()
for f in output.risk_flags.dropna():
    risk_flags.update(f.split(';'))
print(f'Risk flags: {risk_flags}')
