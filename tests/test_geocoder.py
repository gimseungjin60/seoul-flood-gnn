"""실행: python -m tests.test_geocoder  (pytest로도 실행 가능). 실제 API는 호출하지 않는다."""
import pandas as pd

from src.data.geocoder import decide, geocode

DOC = {"address_name": "서울 강남구 역삼동 820-5", "x": "127.03", "y": "37.50"}


def test_decide():
    assert decide([], "강남구")[0] == "NOT_FOUND"
    assert decide([DOC, DOC], "강남구")[0] == "MULTIPLE"
    assert decide([{**DOC, "address_name": "서울 서초구 서초동 1"}], "강남구")[:3] == ("OUT_OF_DISTRICT", None, None)
    assert decide([DOC], "강남구") == ("OK", 37.50, 127.03, "서울 강남구 역삼동 820-5")


def test_geocode_falls_back_to_keyword():
    calls = []

    def fake_search(q, method, key):
        calls.append(method)
        return [] if method == "ADDRESS" else [DOC]

    df = pd.DataFrame({"node_id": ["23-0016"], "district_name": ["강남"], "search_keyword": ["x"],
                       "geocoding_status": ["PENDING"], "geocoding_method": ["NONE"],
                       "latitude": [None], "longitude": [None]})
    r = geocode(df, "k", search=fake_search).iloc[0]
    assert calls == ["ADDRESS", "KEYWORD"]
    assert (r["geocoding_status"], r["geocoding_method"], r["latitude"]) == ("OK", "KEYWORD", 37.50)


if __name__ == "__main__":
    import src.data.geocoder as g
    g.REQUEST_INTERVAL_S = 0
    test_decide()
    test_geocode_falls_back_to_keyword()
    print("ok")
