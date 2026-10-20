import pytest
from fastapi.testclient import TestClient

from app_final import APP_VERSION, app, normalize, store

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_store():
    store.clear()
    yield
    store.clear()


def add_eval(model, reference, hypothesis, **extra):
    response = client.post(
        "/evaluations",
        json={"model": model, "reference": reference, "hypothesis": hypothesis, **extra},
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- service endpoints -------------------------------------------------------

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json()["version"] == APP_VERSION


# --- metrics -----------------------------------------------------------------

def test_normalize_ignores_case_punctuation_and_yo():
    assert normalize("  Ёлка, ЗЕЛЁНАЯ!!  ") == "елка зеленая"


def test_wer_perfect_match_after_normalization():
    response = client.post("/metrics/wer", json={"reference": "Привет, мир!", "hypothesis": "привет мир"})
    assert response.status_code == 200
    body = response.json()
    assert body["wer"] == 0.0
    assert body["cer"] == 0.0


def test_wer_counts_substitution_deletion_insertion():
    # ref: a b c d ; hyp: a x c d e -> 1 substitution (b->x), 1 insertion (e)
    body = client.post("/metrics/wer", json={"reference": "a b c d", "hypothesis": "a x c d e"}).json()
    assert body["substitutions"] == 1
    assert body["insertions"] == 1
    assert body["deletions"] == 0
    assert body["wer"] == 0.5


def test_empty_hypothesis_is_all_deletions():
    body = client.post("/metrics/wer", json={"reference": "раз два три", "hypothesis": ""}).json()
    assert body["deletions"] == 3
    assert body["wer"] == 1.0


def test_reference_of_only_punctuation_is_rejected():
    response = client.post("/metrics/wer", json={"reference": "?!...", "hypothesis": "что"})
    assert response.status_code == 422


def test_too_long_text_is_rejected():
    response = client.post("/metrics/wer", json={"reference": "a" * 5000, "hypothesis": "a"})
    assert response.status_code == 422


# --- evaluations -------------------------------------------------------------

def test_create_and_get_evaluation():
    created = add_eval("whisper-small", "включи свет на кухне", "включи свет в кухне", audio_id="a1.wav")
    assert created["id"] == 1
    assert created["wer"] == 0.25

    response = client.get(f"/evaluations/{created['id']}")
    assert response.status_code == 200
    assert response.json()["audio_id"] == "a1.wav"


def test_get_missing_evaluation_returns_404():
    response = client.get("/evaluations/999")
    assert response.status_code == 404


def test_invalid_model_name_is_rejected():
    response = client.post("/evaluations", json={"model": "bad name!", "reference": "a", "hypothesis": "a"})
    assert response.status_code == 422


def test_list_filters_by_model_and_min_wer():
    add_eval("m1", "раз два", "раз два")
    add_eval("m1", "раз два", "раз")
    add_eval("m2", "раз два", "три")

    assert client.get("/evaluations", params={"model": "m1"}).json()["total"] == 2
    bad = client.get("/evaluations", params={"model": "m1", "min_wer": 0.4}).json()
    assert bad["total"] == 1
    assert bad["items"][0]["wer"] == 0.5


def test_delete_evaluation():
    created = add_eval("m1", "раз", "раз")
    assert client.delete(f"/evaluations/{created['id']}").status_code == 204
    assert client.get(f"/evaluations/{created['id']}").status_code == 404
    assert client.delete(f"/evaluations/{created['id']}").status_code == 404


# --- summaries ---------------------------------------------------------------

def test_model_summary_uses_corpus_wer():
    # 1 error out of 1 word + 0 errors out of 3 words -> corpus WER 0.25, mean WER 0.5
    add_eval("m1", "да", "нет")
    add_eval("m1", "раз два три", "раз два три")
    body = client.get("/models/m1/summary").json()
    assert body["evaluations"] == 2
    assert body["corpus_wer"] == 0.25
    assert body["mean_wer"] == 0.5
    assert body["worst_evaluation_id"] == 1


def test_summary_for_unknown_model_returns_404():
    assert client.get("/models/nope/summary").status_code == 404


def test_leaderboard_ranks_best_model_first():
    add_eval("bad-model", "раз два три", "раз")
    add_eval("good-model", "раз два три", "раз два три")
    board = client.get("/leaderboard").json()
    assert [row["model"] for row in board] == ["good-model", "bad-model"]
    assert board[0]["rank"] == 1
