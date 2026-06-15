from app.agent.state import ExtractionResult, ExtractedEntity, AgentState, Deps


def test_extraction_result_roundtrip():
    er = ExtractionResult(
        patient_name="Jane Doe", patient_age=40, patient_gender="F",
        doc_type="prescription", doc_date="2026-06-10",
        doctor="Dr. Smith",
        diseases=[ExtractedEntity(name="hypertension", confidence=0.9, source_span="Dx: HTN")],
        symptoms=[], medications=[], tests=[],
        confidence=0.8, source_span="full doc",
    )
    assert er.patient_name == "Jane Doe"
    assert er.diseases[0].confidence == 0.9
    assert "hypertension" in er.model_dump_json()


def test_agent_state_is_typeddict():
    st: AgentState = {"messages": [], "intent": None}
    assert st["intent"] is None
