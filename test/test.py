import pytest
from chatbot.src.main import FirstAidChatbot

chatbot = FirstAidChatbot()

test_cases = [
    (
        "I’m sweating, shaky, and my glucometer reads 55 mg/dL—what should I do right now?",
        "Hypoglycemia", "diabetes"
    ),
    (
        "My diabetic father just became unconscious; we think his sugar crashed. What immediate first-aid should we give?",
        "Hypoglycemia", "diabetes"
    ),
    (
        "A pregnant woman with gestational diabetes keeps getting fasting readings around 130 mg/dL. What does this mean and how should we manage it?",
        "Hyperglycemia", "diabetes"
    ),
    (
        "Crushing chest pain shooting down my left arm—do I chew aspirin first or call an ambulance?",
        "Myocardial Infarction", "cardiac"
    ),
    (
        "I’m having angina; how many nitroglycerin tablets can I safely take and when must I stop?",
        "Angina", "cardiac"
    ),
    (
        "Grandma has chronic heart failure, is suddenly short of breath, and her ankles are swelling. Any first-aid steps before we reach the ER?",
        "Heart Failure", "cardiac"
    ),
    (
        "After working in the sun all day I’ve barely urinated and my creatinine just rose 0.4 mg/dL—could this be acute kidney injury and what should I do?",
        "Acute Kidney Injury", "renal"
    ),
    (
        "CKD patient with a potassium level of 6.1 mmol/L—what emergency measures can we start right away?",
        "Hyperkalemia", "renal"
    ),
    (
        "I took ibuprofen for back pain; now my flanks hurt and I’m worried about kidney damage—any immediate precautions?",
        "Acute Kidney Injury", "renal"
    ),
    (
        "Type 2 diabetic, extremely thirsty, glucose meter says ‘HI’ but urine ketone strip is negative—what’s happening and what’s the first-aid?",
        "Hyperglycemia", "diabetes"
    ),
]

@pytest.mark.parametrize("query,expected_condition,expected_domain", test_cases)
def test_chatbot_response(query, expected_condition, expected_domain):
    result = chatbot.process_query(query)
    triage = result['triage']

    assert expected_condition.lower() in triage['condition'].lower()
    assert expected_domain.lower() == triage['domain']
    assert "**Immediate First-Aid Steps:**" in result['response']
    assert "**Sources:**" in result['response']
    assert "⚠️ This information is for educational purposes" in result['response']
    assert any("Local KB" in src.get('source', '') for src in result['sources'])
