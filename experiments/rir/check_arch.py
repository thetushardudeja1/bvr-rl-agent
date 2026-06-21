import sys
sys.path.insert(0, '.')
import stable_baselines3 as sb3
from stable_baselines3 import PPO

m = PPO.load('experiments/rir/bc_model', device='cpu')
me = m.policy.mlp_extractor
print('SB3 version:', sb3.__version__)
print('shared_net attr:', hasattr(me, 'shared_net'))
print('has policy_net:', hasattr(me, 'policy_net'), '| has value_net:', hasattr(me, 'value_net'))
print('share_features_extractor:', m.policy.share_features_extractor)

pn = set(id(p) for p in me.policy_net.parameters())
vn = set(id(p) for p in me.value_net.parameters())
print('policy/value param overlap:', len(pn & vn))

# count params in features_extractor (would be shared if any)
fe = list(m.policy.features_extractor.parameters())
print('features_extractor param tensors:', len(fe))

# full named structure
print('--- top-level policy modules ---')
for name, _ in m.policy.named_children():
    print('  ', name)
