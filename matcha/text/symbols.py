"""Symbol set for Tamil / Tamil–English code-mixed TTS with basic_cleaners.

basic_cleaners lowercases Latin text and leaves Tamil script unchanged.
Update this file if your corpus introduces characters outside this set.
"""

_pad = "_"

# Space kept as an explicit symbol (also used as SPACE_ID)
_space = " "

# Latin letters after basic_cleaners (lowercase only)
_letters_latin = "abcdefghijklmnopqrstuvwxyz"

# Tamil letters, grantha letters used in Tamil, and combining signs
# Covers the full common Tamil orthography (not only chars seen in one corpus)
_letters_tamil = (
    "ஃ"
    "அஆஇஈஉஊஎஏஐஒஓஔ"
    "கஙசஜஞடணதநனபமயரறலளழவஶஷஸஹ"
    "ாிீுூெேைொோௌ்ௗ"
)

# Deduplicate while preserving order
_seen = set()
_symbols = []
for _ch in [_pad, _space] + list(_letters_latin) + list(_letters_tamil):
    if _ch not in _seen:
        _seen.add(_ch)
        _symbols.append(_ch)

symbols = _symbols

# Special symbol ids
SPACE_ID = symbols.index(" ")
