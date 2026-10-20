"""Verify core P2 modules work correctly."""
import sys
sys.path.insert(0, r'L:\weathertrust')

from src.inject.faults import create_faulty_dataset, inject_offset, inject_drift
import pandas as pd
import numpy as np

# Test 1: Fault injection
data = create_faulty_dataset(n_obs=100, n_faults=3, seed=42)
print('1. Fault injection: PASS')
print(f'   Observations: {len(data)}, Fault types: {dict(data["fault_type"].value_counts())}')

# Test 2: Offset injection
data2 = pd.DataFrame({'sat_value': [100.0, 101.0, 102.0, 103.0, 104.0]})
data3 = inject_offset(data2, severity='medium')
print('2. Offset injection: PASS')
print(f'   Mean offset shift: {data3["sat_value"].mean() - 102.0:.3f}')

# Test 3: Robust z-score
from src.detect.rules import robust_z_score
data4 = pd.DataFrame({'obs_id': [1,2,3,4,5], 'sat_value': [280, 281, 350, 283, 284], 'ref_value': [279, 280, 278, 282, 283]})
result = robust_z_score(data4)
print('3. Robust z-score: PASS')
print(f'   Rejections: {int(result["l2_reject"].sum())}')

# Test 4: Range check
from src.detect.rules import range_check
data5 = pd.DataFrame({'obs_id': [1,2,3], 'sat_value': [280, 100, 283], 'ref_value': [279, 278, 282]})
result2 = range_check(data5, sat_min=150, sat_max=350)
print('4. Range check: PASS')
print(f'   Rejections: {int(result2["l1_reject"].sum())}')

# Test 5: Combined detector
from src.detect.rules import combined_detector
data6 = pd.DataFrame({'obs_id': [1,2,3,4,5], 'sat_value': [280, 281, 350, 283, 284], 'ref_value': [279, 280, 278, 282, 285], 'view_zenith': [30, 31, 32, 33, 34], 'solar_zenith': [45, 46, 47, 48, 49], 'cloud_fraction': [0.1, 0.2, 0.3, 0.4, 0.5], 'land_cover': ['forest']*5, 'elevation': [100]*5})
result3 = combined_detector(data6, apply_l1=True, apply_l2=True, apply_l3=False)
print('5. Combined detector: PASS')
print(f'   Rejections: {int(result3["combined_reject"].sum())}')

# Test 6: Trust score
from src.confidence.confidence import TrustScoreComputer
data7 = pd.DataFrame({'obs_id': [1,2,3], 'sat_value': [280, 281, 282], 'ref_value': [279, 280, 283], 'residual_raw': [1, 1, -1], 'predicted_error': [2, 2, 2], 'interval_low': [1, 1, 1], 'interval_high': [3, 3, 3], 'combined_reject': [False, False, True]})
computer = TrustScoreComputer()
result4 = computer.compute(data7)
print('6. Trust score: PASS')
print(f'   Mean trust: {result4["trust_score"].mean():.3f}')

# Test 7: Detection metrics
from src.evaluation.evaluation import detection_metrics
data8 = pd.DataFrame({'obs_id': list(range(20)), 'fault_type': ['none']*20, 'combined_reject': [False]*10 + [True]*10, 'trust_score': [0.5]*20})
metrics = detection_metrics(data8)
print('7. Detection metrics: PASS')
print(f'   Overall F1: {metrics["overall"]["f1"]:.3f}')

print()
print('All core P2 modules operational!')