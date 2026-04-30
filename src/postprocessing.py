"""
Compatibilité : wrapper léger qui réexporte le module de postprocessing
réel situé dans le package `core`.

Le répertoire `src/` est obsolète — pour conserver la compatibilité
avec d'éventuels imports anciens, ce fichier réexporte tout depuis
`core.postprocessing`. Dès que vous supprimez `src/` ce wrapper peut
être supprimé.
"""

from core.postprocessing import *






