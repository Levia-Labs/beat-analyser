# app / static

Browser assets served by Flask. They are separate files because the CSP forbids inline script
and `style=""` ([docs/security.md](../../docs/security.md)); `app.js` also builds the page with
`textContent`, never `innerHTML`. The page that loads them is in [templates](../templates/README.md).

| File      | What it does                                                           |
| --------- | ---------------------------------------------------------------------- |
| `app.js`  | Lists stored folders, uploads for processing, downloads the ZIP result |
| `app.css` | Styles for the page                                                    |
