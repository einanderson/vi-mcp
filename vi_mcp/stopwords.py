"""Stop words dropped from a free-text search query before the full-text index is asked (fts-stopwords, 15.11.2026).

Measured on a production copy (300 queries): natural questions AND all their words, function words included ("wie", "ich", "der",
"how"), so the AND set is empty or wrong and the OR fallback ranks by weak words. Removing them is the lever; no FTS5 tokenizer beat
unicode61 (CHANGELOG 0.67.0). Only the QUERY changes — the index keeps every word, so a quoted search for a stop word still
finds it when the query has nothing else (content_words falls back to the unchanged words).

The lists are the published Snowball stop lists, unchanged, no additions:
  German:  https://snowballstem.org/algorithms/german/stop.txt   (fetched 15.11.2026, sha256 22a7785b53860256…, 231 words)
  English: https://snowballstem.org/algorithms/english/stop.txt  (fetched 15.11.2026, sha256 06b8b14bbe6f6e0b…, 174 words)
"""
# License of the Snowball stop lists (https://snowballstem.org/license.html), BSD-3-Clause:
#   Copyright (c) 2001, Dr Martin Porter, Copyright (c) 2002, Richard Boulton. All rights reserved. Redistribution and use in
#   source and binary forms, with or without modification, are permitted provided that the following conditions are met: 1.
#   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#   disclaimer in the documentation and/or other materials provided with the distribution. 3. Neither the name of the copyright
#   holder nor the names of its contributors may be used to endorse or promote products derived from this software without
#   specific prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
#   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

GERMAN = frozenset("""
aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes anderm andern anderr
anders auch auf aus bei bin bis bist da damit dann der den des dem die das daß derselbe derselben denselben desselben
demselben dieselbe dieselben dasselbe dazu dein deine deinem deinen deiner deines denn derer dessen dich dir du dies
diese diesem diesen dieser dieses doch dort durch ein eine einem einen einer eines einig einige einigem einigen einiger
einiges einmal er ihn ihm es etwas euer eure eurem euren eurer eures für gegen gewesen hab habe haben hat hatte hatten
hier hin hinter ich mich mir ihr ihre ihrem ihren ihrer ihres euch im in indem ins ist jede jedem jeden jeder jedes jene
jenem jenen jener jenes jetzt kann kein keine keinem keinen keiner keines können könnte machen man manche manchem
manchen mancher manches mein meine meinem meinen meiner meines mit muss musste nach nicht nichts noch nun nur ob oder
ohne sehr sein seine seinem seinen seiner seines selbst sich sie ihnen sind so solche solchem solchen solcher solches
soll sollte sondern sonst über um und uns unse unsem unsen unser unses unter viel vom von vor während war waren warst
was weg weil weiter welche welchem welchen welcher welches wenn werde werden wie wieder will wir wird wirst wo wollen
wollte würde würden zu zum zur zwar zwischen
""".split())

ENGLISH = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its
itself they them their theirs themselves what which who whom this that these those am is are was were be been being have
has had having do does did doing would should could ought i'm you're he's she's it's we're they're i've you've we've
they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll we'll they'll isn't aren't wasn't weren't hasn't
haven't hadn't doesn't don't didn't won't wouldn't shan't shouldn't can't cannot couldn't mustn't let's that's who's
what's here's there's when's where's why's how's a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some such no nor not only own same so than too
very
""".split())

STOP = GERMAN | ENGLISH


def content_words(words: list[str]) -> list[str]:
    """The words of a query without stop words (compared in lower case). A query made only of stop words stays as it was — it is
    never emptied, so it still searches for what the person typed."""
    kept = [w for w in words if w.lower() not in STOP]
    return kept or list(words)
