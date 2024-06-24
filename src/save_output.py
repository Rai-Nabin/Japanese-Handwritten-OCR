import json

def save_to_json(data, save_path):
    with open(save_path, 'wb') as fp:
        fp.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    print("Output text saved to JSON.")

