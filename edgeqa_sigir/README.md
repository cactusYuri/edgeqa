# SIGIR Resources Track LaTeX Scaffold (ACM sigconf)

This folder is a minimal ACM `acmart` (sigconf) project intended to help you migrate an existing draft into the SIGIR Resources Track format.

## Quick start

- Main file: `main.tex`
- Sections: `sections/*.tex`
- Bibliography: `references.bib`

If your LaTeX installation has `acmart` installed, you can compile with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use Overleaf and select an ACM SIG Proceedings template.

## Review vs camera-ready

- Review (common): `\documentclass[sigconf,natbib=true,review]{acmart}`
- Double-anonymous (if required): add `anonymous=true`
- Camera-ready: remove `review`/`anonymous=true` and add proper ACM metadata.

Always verify the exact SIGIR 2026 Resources Track CFP requirements.
