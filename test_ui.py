"""Playwright E2E test for the Churn Prediction web UI."""

import threading, time
import uvicorn
from playwright.sync_api import sync_playwright


def test_web_ui():
    """Start server, open web UI, fill form, submit, verify result."""
    from run import app

    t = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": 8006, "log_level": "error"},
        daemon=True,
    )
    t.start()
    time.sleep(4)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto("http://127.0.0.1:8006/", wait_until="networkidle")
        page.wait_for_timeout(500)

        # Fill form
        page.fill('input[name="tenure"]', "12")
        page.fill('input[name="MonthlyCharges"]', "70")
        page.fill('input[name="TotalCharges"]', "840")

        # Submit
        page.click('button[type="submit"]')
        page.wait_for_timeout(2000)

        # Check result is visible
        result = page.locator("#result")
        assert result.is_visible(), "Result card should be visible"

        # Check churn text is shown
        pred_text = page.locator("#predictionText").text_content()
        assert pred_text, "Prediction text should not be empty"
        assert "churn" in pred_text.lower() or "stay" in pred_text.lower()

        # Check probability is shown
        prob_text = page.locator("#probabilityText").text_content()
        assert prob_text, "Probability text should not be empty"
        assert "%" in prob_text

        browser.close()
