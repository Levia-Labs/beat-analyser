# app / templates

`index.html` is the single page of the service. It loads Bootstrap from a CDN and the local
[app.css and app.js](../static/README.md); it holds no inline `<script>` or `style=""`, which
the CSP forbids ([docs/security.md](../../docs/security.md)).
