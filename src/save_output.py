import json

def save_to_json(data):
    with open('output/output.json', 'wb') as fp:
        fp.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    print("Output text saved to JSON.")

