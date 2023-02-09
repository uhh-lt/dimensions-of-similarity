import sys

for fn in sys.argv[1:]:
    lines = open(fn)
    next(lines)
    result_line =  next(lines).strip().split(",")
    _, _, ent, _, _, overall, _, _ = result_line
    _, *dims = result_line
    model_name = fn.split("-eval-")[0].split("-train-")[0].replace("pretrained-", "")
    print(model_name, *[f"{float(d):.02f}" for d in dims], sep=",")
    # print(fn, f"{float(ent):.2f}", f"{float(overall):.2f}")
