import os
import logging
from typing import Any, Dict

from flask import Flask, jsonify, request

# NOTE: module is linage2_service (not lineage2_service)
from linage2_service import (
    load_linage2_bundle,
    process_payload,
)

logger = logging.getLogger("linage2_api")


def _setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_app() -> Flask:
    _setup_logging()

    app = Flask(__name__)

    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    app.config["ARTIFACTS_DIR"] = artifacts_dir

    # Load once at startup
    try:
        logger.info(f"Loading LinAge2 artifacts from: {artifacts_dir!r}")
        app.config["LINAGE2_BUNDLE"] = load_linage2_bundle(artifacts_dir=artifacts_dir)
        logger.info("✓ LinAge2 artifacts loaded")
    except Exception as e:
        logger.exception("✗ Failed to load LinAge2 artifacts")
        app.config["LINAGE2_BUNDLE"] = None
        app.config["LINAGE2_LOAD_ERROR"] = str(e)

    @app.get("/health")
    def health() -> Any:
        bundle = app.config.get("LINAGE2_BUNDLE")
        if bundle is None:
            return jsonify(
                {
                    "status": "degraded",
                    "model_loaded": False,
                    "error": app.config.get("LINAGE2_LOAD_ERROR"),
                }
            ), 503
        return jsonify({"status": "ok", "model_loaded": True}), 200

    @app.post("/predict")
    def predict_endpoint() -> Any:
        """
        LinAge2 prediction endpoint.

        Expected payload:
        {
          "surveys": [{"ques_id":"10740","answer":"8"}, ...],
          "biometrics": {"age":"26","weight":"76","height":"175.0","gender":"1","waist_circumference":"98.0"}
        }

        Response includes:
        - biological_age
        - metadata: chronological_age, delta_ba_ca, per-feature contributions
        """
        logger.info("=" * 60)
        logger.info("Received LinAge2 prediction request")

        payload = request.get_json(silent=True)
        if not payload:
            logger.warning("Invalid or empty JSON payload")
            return jsonify({"code": 400, "biological_age": None, "message": "Invalid JSON payload"}), 400

        bundle = app.config.get("LINAGE2_BUNDLE")
        if bundle is None:
            logger.error("Model bundle not loaded")
            return jsonify({"code": 500, "biological_age": None, "message": "Model not available"}), 500

        result: Dict[str, Any] = process_payload(payload=payload, bundle=bundle)

        if result.get("success"):
            data = result["data"]
            response = {
                "code": 200,
                "biological_age": data.get("biological_age"),
                "message": "Prediction successful",
                "metadata": {
                    "chronological_age": data.get("chronological_age"),
                    "delta_ba_ca": data.get("delta_ba_ca"),
                    "features_used": data.get("features_used"),
                    "total_features": data.get("total_features"),
                    "warnings": result.get("warnings") or None,
                    "feature_contributions": data.get("feature_contributions"),
                    "imputed_features": data.get("imputed_features") or None,
                },
            }
            try:
                logger.info(
                    "✓ Prediction successful: bio=%.2f, chrono=%.2f, delta=%+.2f",
                    float(data["biological_age"]),
                    float(data["chronological_age"]),
                    float(data["delta_ba_ca"]),
                )
            except Exception:
                logger.info("✓ Prediction successful")
            return jsonify(response), 200

        status_code = 400 if result.get("errors") else 500
        response = {
            "code": status_code,
            "biological_age": None,
            "message": "Invalid payload" if status_code == 400 else "Prediction failed",
            "errors": result.get("errors"),
            "warnings": result.get("warnings"),
        }
        logger.error(f"✗ Prediction failed: {response.get('errors')}")
        return jsonify(response), status_code

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
