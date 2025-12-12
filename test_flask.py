import requests
import json

URL = "http://localhost:8000/predict"

payload = {
    "surveys": [
        {"ques_id": "14190", "answer": "0"},
        {"ques_id": "14199", "answer": "0"},
        {"ques_id": "99031", "answer": "3"},
        {"ques_id": "14211", "answer": "1"},
        {"ques_id": "99032", "answer": "3"},
        {"ques_id": "99033", "answer": "1"},

        {"ques_id": "14254", "answer": "1"},
        {"ques_id": "14219", "answer": "1"},
        {"ques_id": "99018", "answer": "1"},
        {"ques_id": "14222", "answer": "1"},
        {"ques_id": "14223", "answer": "0"},
        {"ques_id": "14224", "answer": "1"},
        {"ques_id": "99019", "answer": "0"},
        {"ques_id": "14225", "answer": "0"},
        {"ques_id": "14226", "answer": "1"},
        {"ques_id": "14227", "answer": "1"},
        {"ques_id": "99020", "answer": "1"},
        {"ques_id": "99021", "answer": "0"},
        {"ques_id": "99022", "answer": "1"},
        {"ques_id": "99023", "answer": "1"},
        {"ques_id": "99024", "answer": "1"},
        {"ques_id": "99025", "answer": "1"},
        {"ques_id": "99026", "answer": "1"},
        {"ques_id": "99027", "answer": "1"},
        {"ques_id": "99028", "answer": "1"},
        {"ques_id": "99030", "answer": "1"},

        # --- labs (DB units, reverse-scaled) ---
        {"ques_id": "99001", "answer": "68.0"},
        {"ques_id": "11810", "answer": "111.0"},
        {"ques_id": "11820", "answer": "50.0"},
        {"ques_id": "99002", "answer": "96.9"},
        {"ques_id": "99003", "answer": "18210.0"},
        {"ques_id": "10270", "answer": "110.949"},
        {"ques_id": "99004", "answer": "54.95"},
        {"ques_id": "99005", "answer": "36.2"},
        {"ques_id": "10570", "answer": "81.0"},
        {"ques_id": "10610", "answer": "40.5"},
        {"ques_id": "99006", "answer": "418.45"},
        {"ques_id": "13350", "answer": "2.0"},
        {"ques_id": "13440", "answer": "0.95"},
        {"ques_id": "99016", "answer": "8.0"},
        {"ques_id": "99017", "answer": "1029.0"},
        {"ques_id": "99034", "answer": "3.83"},
        {"ques_id": "99035", "answer": "0.95"},
        {"ques_id": "99036", "answer": "1.027"},
        
    ],
    "biometrics": {
        "age": "72",
        "gender": "1",
        "height": "175.0",
        "weight": "95.86",
        "waist_circumference": ""
    }
}

resp = requests.post(URL, json=payload, timeout=30)

print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
