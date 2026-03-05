"""Tests for cascade output formatting and deterministic ordering."""

from hurricane_debris.models.cascade import Detection, InferenceResult


def test_inference_result_geojson_schema():
    result = InferenceResult(
        image_path="/tmp/example.png",
        width=640,
        height=480,
        detections=[
            Detection(
                bbox=[10, 20, 110, 120],
                category="building_damaged",
                score=0.95,
                priority="critical",
                mask=None,
            )
        ],
    )

    geo = result.to_geojson()
    assert geo["type"] == "FeatureCollection"
    assert geo["coordinate_system"] == "image_pixel"
    assert geo["image_size"]["width"] == 640
    assert len(geo["features"]) == 1

    feature = geo["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["bbox_xyxy"] == [10.0, 20.0, 110.0, 120.0]


def test_detection_priority_sort_key_is_deterministic():
    detections = [
        Detection([0, 0, 10, 10], "vehicle", 0.50, None, "high"),
        Detection([0, 0, 9, 9], "vehicle", 0.90, None, "high"),
        Detection([0, 0, 8, 8], "building_damaged", 0.40, None, "critical"),
        Detection([0, 0, 7, 7], "road_no_damage", 0.60, None, "low"),
    ]

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    detections.sort(
        key=lambda d: (
            priority_order.get(d.priority, 9),
            -float(d.score),
            d.category,
            tuple(float(v) for v in d.bbox),
        )
    )

    assert [d.priority for d in detections] == ["critical", "high", "high", "low"]
    assert detections[1].score >= detections[2].score


def test_inference_result_json_schema_like_validation():
    result = InferenceResult(
        image_path="/tmp/example.png",
        width=100,
        height=120,
        detections=[
            Detection(
                bbox=[1, 2, 30, 40],
                category="vehicle",
                score=0.8,
                priority="high",
                mask=None,
            )
        ],
    )

    payload = result.to_json()
    assert set(payload.keys()) == {"image", "width", "height", "num_detections", "detections"}
    assert payload["num_detections"] == 1
    assert isinstance(payload["detections"], list)
    det = payload["detections"][0]
    assert set(det.keys()) == {"bbox", "category", "score", "priority", "mask_available"}
    assert isinstance(det["mask_available"], bool)
