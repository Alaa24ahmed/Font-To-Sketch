import vharfbuzz as hv

animal_names = [
  {"english": "cat", "arabic": "قطة"},
  {"english": "Lion", "arabic": "أسد"},
  {"english": "Elephant", "arabic": "فيل"},
  {"english": "Tiger", "arabic": "نمر"},
  {"english": "Cheetah", "arabic": "فهد"},
  {"english": "Monkey", "arabic": "قرد"},
  {"english": "Dolphin", "arabic": "دلفين"},
  {"english": "Penguin", "arabic": "بطريق"},
  {"english": "Kangaroo", "arabic": "كنغر"},
  {"english": "Fox", "arabic": "ثعلب"},
  {"english": "Eagle", "arabic": "نسر"},
  {"english": "Wolf", "arabic": "ذئب"},
  {"english": "Turtle", "arabic": "سلحفاة"},
  {"english": "Panda", "arabic": "باندا"},
  {"english": "Giraffe", "arabic": "زرافة"},
  {"english": "Bear", "arabic": "دب"},
  {"english": "Owl", "arabic": "بومة"}
]

fontpath = './data/fonts/ArefRuqaa.ttf'
vhb = hv.Vharfbuzz(fontpath)

path_templ = "/Users/bkhmsi/Desktop/Animal-Words/correct/{}.svg"

for animal in animal_names:
    txt = animal["arabic"]
    buf = vhb.shape(txt, {"features": {"kern": True, "liga": True}})
    svg = vhb.buf_to_svg(buf)
    with open(path_templ.format(animal["english"]), 'w') as fout:
        fout.write(svg)
