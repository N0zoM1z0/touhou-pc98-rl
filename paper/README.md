# Research manuscript

`main.tex` is a living English research paper for the project. It separates
measured results from hypotheses and should be updated whenever an experiment
changes the evidence.

Build it from the repository root:

```bash
make paper
```

The intermediate PDF is written to ignored `paper/build/main.pdf`, then copied
to tracked `paper/main.pdf` as the public research snapshot.
