# Claude Guidelines for SupportModel Project

## Critical Data Integrity Rules

### Results File Contract
- **DO NOT change the results file generated in the Colab** - The results file is used in the Streamlit app and represents a strict contract between the evaluation system and the UI
- Any changes to the results file structure must be coordinated with corresponding Streamlit app updates
- The results file format serves as the interface specification between data generation (Colab) and data visualization (Streamlit)

### Metrics Authenticity
- **ALWAYS use only real metrics, no random, simulated or invented numbers**
- All evaluation metrics must be calculated using actual model outputs, real embeddings, and genuine API responses
- If simulated metrics need to be created for any reason (testing, debugging, etc.), this must be **EXPLICITLY marked and disclosed** in the Streamlit app interface
- Users must be clearly informed when viewing simulated vs. real data

## Implementation Guidelines

### When Working with Evaluation Code
- Verify that all metrics calculations use real model inference
- Ensure API calls (OpenAI, sentence-transformers, etc.) are genuine
- Avoid placeholder values or mock responses
- Document any fallback mechanisms clearly

### When Working with Streamlit UI
- Display data source clearly (real vs. simulated)
- Add verification indicators for data authenticity
- Include metadata about evaluation methods used
- Provide transparency about metric calculation approaches

### File Modification Protocol
- Colab notebooks: Only fix bugs, never change metric calculation logic
- Streamlit apps: Can be modified to better display existing data
- Results files: Never modify directly - regenerate if needed
- Configuration files: Changes allowed but must maintain backward compatibility

## Quality Assurance
- All metrics must trace back to actual model performance
- Cross-reference results with evaluation methodology
- Validate that reported numbers match actual system capabilities
- Maintain audit trail of evaluation processes