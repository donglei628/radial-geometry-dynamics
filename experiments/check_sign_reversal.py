import json
import numpy as np

f = open(r'C:\source\NextTransformer\paper2\experiments\results\e5_1_rho_profiles.json')
data = json.load(f)
f.close()

print("=== SIGN REVERSAL ONSET ===")
for model_key in ['tinyllama', 'qwen2.5-1.5b']:
    m = data[model_key]
    layers = m['layers']
    n = len(layers)
    print("%s (%d layers):" % (model_key, n))
    first_neg = None
    for l in layers:
        if l['rho_up_down'] < 0 and first_neg is None:
            first_neg = l['layer']
    if first_neg is not None:
        pct = first_neg / n * 100
        print("  First negative at layer %d (%.0f%% depth)" % (first_neg, pct))
    # Show last few layers
    for l in layers[-8:]:
        print("    L%d: rho_up_down=%.4f" % (l['layer'], l['rho_up_down']))

# Check TinyLlama sign reversal onset
print("")
tl = data['tinyllama']
layers = tl['layers']
n = len(layers)
print("TinyLlama (%d layers) - last layers with negative rho:" % n)
for l in layers:
    if l['rho_up_down'] < 0:
        print("  Layer %d: rho=%.4f" % (l['layer'], l['rho_up_down']))

# Check Pythia from separate file
print("")
print("=== Checking Pythia-1B rho from e5_1 ===")
if 'pythia-1b' in data:
    p = data['pythia-1b']
    layers = p['layers']
    n = len(layers)
    print("Pythia-1B (%d layers):" % n)
    for l in layers:
        if l.get('rho_up_down', l.get('rho_w1_w2', 0)) < 0:
            rho_val = l.get('rho_up_down', l.get('rho_w1_w2', 0))
            print("  Layer %d: rho=%.4f" % (l['layer'], rho_val))
