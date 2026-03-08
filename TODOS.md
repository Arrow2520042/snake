# Snake Project TODOs

## Roadmap (8 punktow)

- [ ] Rozbic `game.py` na moduly: `ui.py`, `renderer.py`, `live_trainer.py`, `designer.py`.
- [ ] Dodac panel metryk treningu online: srednia nagroda z N epizodow, dlugosc epizodu, win-rate.
- [ ] Dodac tryb ewaluacji checkpointu bez uczenia (epsilon=0) i porownanie modeli.
- [ ] Zapisywac metadane checkpointu (`algo`, `eps`, `gamma`, `level`, `date`) obok `.pth`/`.npy`.
- [ ] Dodac testy jednostkowe dla: kolizji, state encoding, mapowania akcji, load/save poziomow.
- [ ] Dodac replay viewer (odtwarzanie epizodu z logu).
- [ ] Dodac obsluge seedow i tryb deterministyczny dla replikowalnych eksperymentow.
- [ ] Dodac profiler FPS/krokow i benchmark `dqn` vs `qlearning`.
