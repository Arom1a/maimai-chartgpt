from bidict import frozenbidict


tokens = frozenbidict(
    {
        "<SOS>": 1,
        "<EOS>": 2,
    }
)

print(tokens["<SOS>"])
print(tokens.inv[2])
