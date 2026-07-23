# ACM template provenance

The paper is built with the ACM production `acmart` v2.19 release dated
2026-06-27.

The ICAIF call links the ACM primary template ZIP at
<https://portalparts.acm.org/hippo/latex_templates/acmart-primary.zip>. On
2026-07-24 that endpoint returned HTTP 403 to this non-browser build
environment. The matching production package was therefore downloaded from
the CTAN distribution endpoint explicitly identified by the package README as
one of the two production sources:
<https://mirrors.ctan.org/macros/latex/contrib/acmart.zip>.

Downloaded ZIP SHA-256:
`5011d4a698b993ffd4c9a6a24cf4cde7dc9deb4a7ec2dc357dbbe4ec15085618`.

The files in `acmart/` are unmodified upstream artifacts. `acmart.cls` was
generated from `acmart.dtx` with `latex acmart.ins`. They are retained together
because the generated class requires the source to accompany its distribution.
The source is licensed under the LaTeX Project Public License; the bibliography
style declares itself public domain.

| File | SHA-256 |
|---|---|
| `acmart.cls` | `2f949e6e3f2a79f2cdc218b9dcdbaa7dd451adb4ee0be1af6dc7ebe00b318ea7` |
| `ACM-Reference-Format.bst` | `8ec002c927068bfc5b3cfe71b66aa4767b9e485530ac3c67ba5c064df4c2e6ac` |
| `acmart.dtx` | `4e2fd29fc8a45c3d02facb616ed29954d7932891883185460bf374303036f3e1` |
| `acmart.ins` | `fa7daaa9375775ce699f33f909667bdb83c0408fd8d67606926df86c346c4044` |

The local `.latexmkrc` prepends this directory to both `TEXINPUTS` and
`BSTINPUTS`, ensuring that the vendored v2.19 class and bibliography style are
used instead of the host TeX Live v1.81 class.
