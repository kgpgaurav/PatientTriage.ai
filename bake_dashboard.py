import json
import os

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
DATA_PATH = os.path.join(os.path.dirname(__file__), "outputs", "dashboard_data.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "dashboard.html")


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit("outputs/dashboard_data.json not found — run simulate.py first")

    with open(DATA_PATH) as f:
        data = f.read()
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    if "__DATA__" not in template:
        raise SystemExit("dashboard_template.html is missing the __DATA__ placeholder")

    out = template.replace("__DATA__", data)
    with open(OUT_PATH, "w") as f:
        f.write(out)

    print(f"wrote {OUT_PATH} ({len(out)} bytes)")


if __name__ == "__main__":
    main()
