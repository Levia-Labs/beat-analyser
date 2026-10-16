# tasks / docker

Namespace `docker:`. The container is defined by `docker-compose.yml` at the repository root
and built from `app/Dockerfile`.

| File | Task | Role |
| --- | --- | --- |
| `hub.md` | - | The Docker Hub page of `levialabs/beat-analyser`, written for people pulling the image |
| `hub.py` | `task docker:hub:check` | Compare the live page with `hub.md`. Reads only |
| `hub.py` | `task docker:hub:publish` | Publish `hub.md` and the short description in `hub.py`. Signs in with the `docker login` credentials `docker push` uses, or `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` when set. Token scope: Read, Write, Delete |

`docker:hub:publish` and `docker push` change a public page and are denied to agents in
`.claude/settings.json`; a person runs them. Keep `hub.md` in step with `README.md` and
`docs/output.md` when the API or the output format changes.
