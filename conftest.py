# Present so pytest puts the repo root on sys.path and `import transcribe` works from
# tests/. Deliberately empty otherwise: no fixture here should be able to reach a model,
# the network, or an audio file — every test in this repo runs on plain data.
