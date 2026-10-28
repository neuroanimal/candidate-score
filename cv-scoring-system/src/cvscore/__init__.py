"""cvscore - explainable CV, cover letter and job-offer scoring without LLMs.

Pipeline: extraction (PDF text layer / OCR, DOCX, TXT) -> structure (sections,
timeline, contact) -> job-offer parsing (requirements, knock-outs) -> about 60
metrics in four groups -> composite scores with confidence -> recommendations ranked
by their effect on the score -> HTML/Markdown/JSON reports.
"""

__version__ = "0.1.0"
