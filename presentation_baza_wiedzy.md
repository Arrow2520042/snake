# Baza wiedzy (szczegoly do prezentacji)

Ten plik rozwija kazdy punkt z prezentacji, w tej samej kolejnosci.
Gdy pojawia sie wzor, ponizej znajduje sie wyjasnienie wszystkich symboli, w tym liter greckich.

## Architektura
1) Rownolegle dziala wiele srodowisk, aby szybciej zbierac dane i zwiekszac roznorodnosc sytuacji uczacych.
	Taki uklad poprawia stabilnosc, bo uczenie nie opiera sie na jednym epizodzie.
2) Dla kazdego stanu wyznaczana jest maska bezpiecznych akcji, ktora eliminuje ruchy z natychmiastowa kolizja.
	To ogranicza liczbe bezsensownych krokow i skraca czas uczenia podstawowych zasad bezpieczenstwa.
3) Model z rozdzieleniem wartosci stanu i przewagi akcji w sieci konwolucyjnej wylicza wartosci funkcji Q dla calej paczki stanow.
	W praktyce siec ma dwa tory: wartosc stanu (jak dobry jest stan niezaleznie od akcji) oraz przewage akcji (ktora akcja jest lepsza od innych).
	Dopiero polaczenie tych dwoch strumieni daje ostateczna ocene akcji w danym stanie.
	Dzieki temu siec potrafi oddzielic sytuacje, gdy wszystkie akcje sa slabe, od sytuacji, gdy jedna akcja wyraznie dominuje.
4) Agent wybiera akcje strategia epsilon-zachlanna z uwzglednieniem maski, laczac eksploracje z eksploatacja.
	Gdy epsilon jest wysoki, agent testuje nowe zachowania, a gdy niski, wybiera najlepiej oceniane akcje.
5) Funkcja kroku srodowiska zwraca przejscie (stan, akcja, nagroda, nowy stan, zakonczony epizod).
	Informacja o zakonczonym epizodzie jest kluczowa, aby nie liczyc dalszych nagrod po koncu gry.
6) Przejscia trafiaja do bufora zwrotu wielokrokowego, a nastepnie do pamieci z priorytetami.
	Najpierw scalane sa nagrody z kilku krokow, a dopiero potem przejscie jest zapisywane w pamieci z priorytetami.
	Bufor jest wykorzystywany podczas kroku uczenia (aktualizacji), gdy agent probkuje paczke przejsc w `CNNAgent.update()`.
	Wtedy pamiec z priorytetami narzuca czestsze wybieranie przejsc z duzym bledem i nadaje im wieksza wage.
	Efekty pracy pamieci z priorytetami widac, gdy trudne lub negatywne przejscia wracaja czesciej do uczenia i szybciej koryguja zachowanie agenta.
	W praktyce oznacza to szybszy spadek powtarzalnych kolizji i stabilniejszy wzrost wynikow.
7) Aktualizacja modelu polega na probkowaniu paczki, obliczeniu straty i kroku optymalizacji.
	Po aktualizacji wagi priorytetow sa odswiezane na podstawie nowych bledow uczenia.
8) Siec docelowa jest aktualizowana przez lagodne mieszanie wag, aby unikac niestabilnych skokow.
	Taki mechanizm ogranicza wahania wartosci Q i zmniejsza ryzyko rozjechania uczenia.
9) Harmonogram zmniejsza wspolczynnik uczenia, a przy spadku wyniku nastepuje powrot do najlepszego punktu kontrolnego.
	Po cofnieciu resetowana jest pamiec doswiadczen i zwiekszana jest eksploracja, aby uniknac zlej lokalnej polityki.

## Wejscia i wyjscia
1) Wejscie to stan siatki i wektor cech pomocniczych, a wyjscie to wartosci Q dla trzech akcji.
	Wartosci Q opisuja oczekiwana sume przyszlych nagrod dla kazdej akcji.
2) Akcje odpowiadaja ruchom wzgledem aktualnego kierunku: prosto, w prawo, w lewo.
	Taki opis jest stabilny niezaleznie od orientacji na planszy.
3) Rozmiar wejscia wynika z czterech kanalow planszy i pieciu cech pomocniczych.
	Cztery kanaly opisuje sytuacje na siatce, a piec cech streszcza informacje kierunkowe i dlugosc weza.
4) Strategia epsilon-zachlanna wybiera akcje z najwyzsza wartoscia Q lub losowo.
	Dzieki temu agent nie zamyka sie zbyt wczesnie w jednej strategii.

## Obserwacja dla sieci konwolucyjnej
1) Cztery kanaly siatki koduja wiek ciala, pozycje glowy, pozycje jedzenia oraz sciany.
	Wiek ciala daje informacje o kierunku ruchu i o tym, gdzie jest ogon.
2) Wektor cech pomocniczych zawiera cztery kierunki (kod one-hot) oraz znormalizowana dlugosc weza.
	Normalizacja utrzymuje skale cech na podobnym poziomie, co ulatwia uczenie.
3) Tensor ma ksztalt 4 x rozmiar_planszy x rozmiar_planszy, po czym jest splaszczany i laczony z cechami pomocniczymi.
	Tensor to wielowymiarowa tablica, ktora zachowuje uklad przestrzenny danych z planszy.
	Jest uzyty, bo siec konwolucyjna uczy sie wzorcow przestrzennych, takich jak ksztalt ciala, bariery i odleglosc do jedzenia.
	Splaszczenie oznacza zamiane wielu wymiarow na jeden wektor cech, aby przekazac dane do warstw w pelni polaczonych.
	Cechy pomocnicze sa zamieniane na wektor: kierunek to kod one-hot (cztery pozycje 0/1), a dlugosc weza jest normalizowana do jednej liczby.
	Na koncu wektor z konwolucji jest konkatenowany z wektorem cech pomocniczych, tworzac jedno wejscie dla czesci gleboko polaczonej.

## Maskowanie akcji
1) Maskowanie usuwa akcje prowadzace do natychmiastowej kolizji ze sciana lub wlasnym cialem.
	To ogranicza liczbe krotkich epizodow, w ktorych agent umiera bez nauki.
2) Przy duzej zajetosci planszy dodawane jest krotkie przewidywanie kilku ruchow do przodu.
	Pozwala to wykryc ruchy pozornie bezpieczne, ktore w kolejnym kroku prowadza do pulapki.
3) Gdy wszystkie akcje sa zle, maska wraca do [True, True, True].
	Zapobiega to zatrzymaniu decyzji i pozwala agentowi wykonac jakikolwiek ruch.

## Pamiec z priorytetami i zwrot wielokrokowy
1) Najpierw zbierane sa zwykle przejscia krok po kroku w kazdym srodowisku.
	Kazdy krok daje stan, akcje, nagrode, nowy stan i informacje o koncu epizodu.
2) Bufor zwrotu wielokrokowego gromadzi kilka krokow i scala nagrody w jeden zwrot.
	To powoduje, ze sygnal nagrody szybciej dociera do decyzji z poczatku sekwencji.
3) Gdy bufor uzbiera wystarczajaco krokow (albo epizod sie konczy), tworzony jest rekord wielokrokowy.
	Rekord zawiera stan startowy, akcje, skumulowana nagrode, nowy stan i liczbe krokow.
4) Rekord wielokrokowy trafia do pamieci z priorytetami z priorytetem startowym.
	Dzieki temu nowe i potencjalnie wazne przejscia maja szanse byc szybko uzyte w uczeniu.
5) Podczas uczenia probkowana jest paczka z prawdopodobienstwem proporcjonalnym do priorytetu.
	To sprawia, ze trudne przejscia sa widziane czesciej niz latwe.
6) Probkowanie jest realizowane segmentami: suma priorytetow jest dzielona na przedzialy, a z kazdego losuje sie wartosc.
	Nastepnie SumTree zwraca indeksy i dane dla wylosowanych wartosci, co daje zbalansowany rozklad probek.
7) Po probkowaniu liczone sa wagi korekcyjne, aby skorygowac stronniczosc probkowania.
	Wagi rosna, gdy przejscie jest rzadkie, i maleja, gdy jest nadreprezentowane.
8) Po obliczeniu bledu roznicy czasowej priorytety sa aktualizowane.
	Jesli blad jest duzy, przejscie pozostaje wazne i wraca do uczenia czesciej.
9) Negatywne doswiadczenia (kary, kolizje, brak postepu) daja duze bledy.
	Taki blad zwieksza priorytet, wiec model szybciej uczy sie unikac tych sytuacji.

Definicje podstawowe:
- Blad roznicy czasowej dla i-tego przejscia to $\delta_i = Q(s,a) - y$.
  To roznica miedzy tym, co siec przewiduje dla akcji $a$ w stanie $s$, a celem $y$.
- Zwrot wielokrokowy to suma zdyskontowanych nagrod z kilku kolejnych krokow, aby szybciej przenosic sygnal nagrody.
- Wspolczynnik dyskonta przyszlych nagrod $\gamma$ (gamma) okresla, jak bardzo cenimy przyszle nagrody.
  Im mniejsze $\gamma$, tym bardziej licza sie nagrody natychmiastowe; im wieksze, tym bardziej licza sie nagrody odlegle.

Na czym polega probkowanie i jak aktualizowane sa wagi:
- SumTree przechowuje priorytety w lisciach i sumy priorytetow w wezlach wewnetrznych.
- Probkowanie dzieli sume priorytetow na segmenty i losuje jedna wartosc z kazdego segmentu.
- Dla kazdej wylosowanej wartosci SumTree zwraca indeks liscia, priorytet i przejscie.
- Z priorytetow obliczane sa prawdopodobienstwa $P(i)$ i wagi korekcyjne $w_i$.
- Wagi $w_i$ sa uzywane w funkcji straty jako mnozniki, ale nie sa zapisywane w drzewie.
- Po obliczeniu bledow $\delta_i$ priorytety sa aktualizowane przez `update_priorities`.

Batch update i zwiazek z SumTree:
- `batch_update` dostaje tablice indeksow lisci i nowe priorytety obliczone z bledow.
- Dla kazdego liscia liczy sie zmiana: change = nowy_priorytet - stary_priorytet.
- Zmiana jest propagowana w gore drzewa, aby odzwierciedlic nowa sume priorytetow.
- To zapewnia, ze kolejne probkowanie od razu uwzglednia nowy rozklad waznosci przejsc.

SumTree i batch_update (intuicja i matematyka):
- SumTree to binarne drzewo sum, gdzie kazdy wezel przechowuje sume priorytetow swoich dzieci.
- Lisc reprezentuje pojedyncze przejscie i jego priorytet.
- Losowanie polega na wylosowaniu wartosci v z przedzialu [0, suma] i zejscie w dol drzewa:
  jesli v <= suma lewego dziecka, idziemy w lewo; w przeciwnym razie v = v - suma lewego dziecka i idziemy w prawo.
- Zmiana priorytetu w batch_update jest propagowana w gore drzewa:
  change = nowy_priorytet - stary_priorytet, a nastepnie kazdy przodek dostaje tree[parent] += change.
- Zlozonosc aktualizacji pojedynczego liscia to O(log N), a batch_update realizuje to wielokrotnie dla calej paczki.

Gdzie bufor jest zapisywany:
- Bufor z priorytetami jest trzymany w pamieci RAM (SumTree i dane przejsc) i sluzy tylko w trakcie sesji uczenia.
- Nie jest zapisywany w checkpointach modelu, wiec po restarcie treningu bufor startuje pusty.
- Po rollbacku bufor jest resetowany, zeby nie uczyc sie na starych, mniej aktualnych doswiadczeniach.

Wzor priorytetu:
$p_i = (|\delta_i| + \epsilon)^{\alpha}$
Wyjasnienie symboli:
- $p_i$ to priorytet i-tego przejscia w buforze.
- $\delta_i$ to blad roznicy czasowej dla i-tego przejscia.
- $\epsilon$ to mala stala stabilizujaca, aby priorytet nie zerowal sie przy malym bledzie.
- $\alpha$ (alfa) kontroluje jak silnie priorytet zalezy od bledu.

Wzor prawdopodobienstwa probkowania:
$P(i) = p_i / \sum_j p_j$
Wyjasnienie symboli:
- $P(i)$ to prawdopodobienstwo wylosowania i-tego przejscia.
- $\sum_j p_j$ to suma priorytetow wszystkich przejsc.
- $j$ to indeks po wszystkich elementach bufora.

Wzor wag korekcyjnych:
$w_i = (N * P(i))^{-\beta}$
Wyjasnienie symboli:
- $w_i$ to waga korekcyjna dla i-tego przejscia.
- $N$ to liczba przejsc w buforze.
- $P(i)$ to prawdopodobienstwo probkowania.
- $\beta$ (beta) kontroluje sile korekty stronniczosci probkowania.

Wzor zwrotu wielokrokowego:
$R^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k}$
Wyjasnienie symboli:
- $R^{(n)}$ to zwrot wielokrokowy z n kolejnych krokow.
- $n$ to liczba krokow w zwrocie.
- $\gamma$ (gamma) to wspolczynnik dyskonta przyszlych nagrod.
- $r_{t+k}$ to nagroda w kroku $t+k$.
- $k$ to indeks kroku w sumie.

## Strata i aktualizacja
1) Najpierw probkowana jest paczka przejsc z pamieci z priorytetami, razem z wagami korekcyjnymi.
	Paczka zawiera stany, akcje, nagrody, nowe stany, informacje o koncu epizodu i liczbe krokow.
2) Dla kazdego przejscia liczona jest przewidywana wartosc funkcji Q dla wykonanej akcji.
	To jest to, co siec uczona obecnie sadzi o jakosci tej akcji.
3) Cel jest liczony z oddzielna siecia docelowa, aby rozdzielic wybor akcji i jej wycene.
	Akcje wybiera siec uczona (argmax), a wartosc tej akcji ocenia siec docelowa.
4) To oznacza, ze polityka docelowa jest zachlanna wzgledem sieci uczonej, ale stabilizowana przez siec docelowa.
	Siec docelowa nie ustala zachowania, tylko dostarcza stabilnej wyceny celu uczenia.
5) Funkcja straty uwzglednia wagi z pamieci z priorytetami, aby skorygowac stronniczosc probkowania.
	Wyzszy priorytet oznacza wyzszy wplyw danego przejscia na aktualizacje wag.
6) Po obliczeniu straty wykonywany jest krok optymalizatora, a nastepnie aktualizowane sa priorytety.
	Bledy roznicy czasowej wyznaczaja, ktore przejscia powinny wracac do uczenia.
7) Negatywne doswiadczenia obciazaja cel i zwiekszaja blad, wiec gradienty obnizaja wartosci Q dla zlych akcji.
	To bezposrednio przeklada sie na spadek atrakcyjnosci niebezpiecznych decyzji.
8) Siec docelowa jest aktualizowana lagodnie przez mieszanie wag, co stabilizuje uczenie.
	Wspolczynnik mieszania jest niewielki, dzieki czemu siec docelowa zmienia sie powoli.

Proces probkowania paczki i aktualizacji wag (krok po kroku):
1) Pamiec z priorytetami zwraca paczke przejsc oraz wagi korekcyjne.
2) Dane sa zamieniane na tensory i trafiaja na urzadzenie obliczeniowe.
3) Siec uczona wyznacza Q dla wybranych akcji, a siec docelowa wyznacza cel.
4) Liczona jest strata z wagami korekcyjnymi, po czym wykonywany jest krok optymalizatora.
5) Bledy roznicy czasowej aktualizuja priorytety, aby trudne przejscia wracaly czesciej.

Wyjasnienie gradientow:
- Gradient to pochodna funkcji straty wzgledem wag sieci.
- Backpropagation liczy te pochodne warstwa po warstwie i wskazuje, jak zmienic wagi, aby zmniejszyc strate.
- Im wiekszy blad, tym wiekszy gradient i silniejsza korekta wag.

Skad bierze sie siec docelowa:
- Na starcie treningu siec docelowa jest kopia sieci uczonej (identyczne wagi).
- Pozniej jest aktualizowana lagodnie przez Polyak averaging, zamiast uczenia bezposrednio z gradientow.
- Dzieki temu cele uczenia sa stabilniejsze niz gdyby obie sieci zmienialy sie jednoczesnie.

Wyjasnienie Polyak averaging (lagodnej aktualizacji):
- Wzor: $\theta_{docelowa} \leftarrow \tau\,\theta_{uczona} + (1-\tau)\,\theta_{docelowa}$.
- $\tau$ (tau) to wspolczynnik mieszania; mala wartosc oznacza powolne podazanie sieci docelowej za uczona.
- Dzieki temu cele uczenia zmieniaja sie plynie, co redukuje oscylacje.
 - W praktyce kazda waga sieci docelowej jest przesuwana o maly ulamek w kierunku wagi sieci uczonej.
 - To dziala jak filtr dolnoprzepustowy: siec docelowa jest "gladsza" i mniej reaguje na chwilowy szum w aktualizacjach.
 - Gdy $\tau$ jest bardzo male, siec docelowa zmienia sie wolno; gdy wieksze, szybciej nadaza za siecia uczona.

Wyjasnienie optymalizatora Adam:
- Adam utrzymuje ruchome srednie pierwszego i drugiego momentu gradientu (m i v).
- Daje to adaptacyjny krok uczenia dla kazdego parametru, wiekszy dla rzadkich gradientow i mniejszy dla szumiacych.
- Zwykle stosuje wspolczynniki wygaszania $\beta_1$ i $\beta_2$ (beta) oraz mala stala $\epsilon$ dla stabilnosci numerycznej.
- Efekt: szybsza i bardziej stabilna konwergencja niz przy stalych krokach uczenia.

Wzor celu:
$y = r + \gamma^n Q_{sieci\_docelowej}(s', arg\,max Q_{sieci\_uczonej})$
Wyjasnienie symboli:
- $y$ to wartosc docelowa dla aktualizacji.
- $r$ to natychmiastowa nagroda po wykonaniu akcji.
- $\gamma$ (gamma) to wspolczynnik dyskonta.
- $n$ to liczba krokow w zwrocie wielokrokowym.
- $Q_{sieci\_docelowej}$ to wartosc Q wyliczona przez siec docelowa.
- $Q_{sieci\_uczonej}$ to wartosc Q wyliczona przez siec uczona.
- $s'$ to nowy stan po wykonaniu akcji.
- $arg\,max$ oznacza argument maksimum, czyli akcje o najwyzszej wartosci Q.

Wyjasnienie celu $y$ (intuicja):
- $y$ to "oczekiwana" wartosc, do ktorej dopasowujemy $Q(s,a)$.
- Sklada sie z dwoch czesci: natychmiastowej nagrody $r$ oraz zdyskontowanej oceny tego, co najlepsze dalej.
- Siec uczona wybiera najlepsza akcje w nowym stanie ($arg\,max$), a siec docelowa ocenia jej jakosc.
- Dzieki temu uczymy sie przewidywac nie tylko to, co dostalismy teraz, ale tez to, co obiecuje przyszlosc.

Skad sie bierze oczekiwana wartosc $y$ i co reprezentuje:
- $y$ powstaje bezposrednio z definicji uczenia Q: bierzemy realna nagrode $r$ i dodajemy przewidywana "reszte" przyszlych nagrod.
- Ta przewidywana reszta to wartosc Q nowego stanu i najlepszej akcji, zdyskontowana przez $\gamma^n$.
- W praktyce $y$ jest celem uczenia dla pary (stan, akcja): mowi, ile lacznie ta akcja "powinna" byc warta.
- Dopasowanie polega na minimalizacji roznicy miedzy $Q(s,a)$ a $y$ w funkcji straty (np. MSE).
- Gdy $Q(s,a)$ jest ponizej $y$, gradienty podnosza wartosc tej akcji; gdy jest powyzej, gradienty ja obnizaja.

Wzor funkcji straty:
$L = E[w * (Q - y)^2]$
Wyjasnienie symboli:
- $L$ to wartosc straty.
- $E$ to srednia po probce uczacej.
- $w$ to waga korekcyjna z pamieci z priorytetami.
- $Q$ to przewidywana wartosc Q dla wybranej akcji.
- $y$ to wartosc docelowa.

Wzor lagodnej aktualizacji wag:
$\theta_{docelowa} \leftarrow \tau\,\theta_{uczona} + (1-\tau)\,\theta_{docelowa}$
Wyjasnienie symboli:
- $\theta_{docelowa}$ to wagi sieci docelowej.
- $\theta_{uczona}$ to wagi sieci uczonej.
- $\tau$ (tau) to wspolczynnik mieszania wag.

## Harmonogram wspolczynnika uczenia i cofanie
1) Harmonogram zmniejsza wspolczynnik uczenia, gdy metryka nie poprawia sie przez dluzszy czas.
	Zapobiega to sytuacji, w ktorej zbyt duzy krok uczenia rozbija stabilnosc.
2) Cofanie laduje najlepszy punkt kontrolny i resetuje pamiec doswiadczen.
	Dzieki temu uczenie wraca do sprawdzonego stanu i buduje nowe doswiadczenia.
3) Po cofnieciu zwiekszana jest eksploracja i resetowane sa parametry uczenia.
	To daje agentowi szanse na wyjscie z niekorzystnej strategii.

## Ksztaltowanie nagrody
1) Tryby nagrody dziela sie na prosta nagrode, zlozona nagrode oraz mieszanie zalezne od zajetosci planszy.
	Mieszanie pozwala plynnie przechodzic do bardziej zlozonych kryteriow, gdy gra staje sie trudniejsza.
2) Wspolczynnik zajetosci opisuje, jak duza czesc planszy jest zajeta przez weza.
	To kluczowy sygnal, bo wraz ze wzrostem zajetosci rosnie ryzyko pulapek.
3) Waga mieszania oblicza, jak duzy udzial maja skladniki proste i zlozone.
	Waga jest obcinana do zakresu [0, 1], aby uniknac wartosci ujemnych lub zbyt duzych.
4) Nagroda mieszana jest suma wazona czesci prostej i zlozonej.
	Ten mechanizm pozwala zachowac stabilnosc wczesnego treningu i wiecej finezji pozniej.

Wzor zajetosci:
$occ = dlugosc\_weza / (rozmiar\_planszy * rozmiar\_planszy)$
Wyjasnienie symboli:
- $occ$ to wspolczynnik zajetosci planszy.
- $dlugosc\_weza$ to liczba segmentow weza.
- $rozmiar\_planszy$ to liczba pol w jednym wymiarze planszy.

Wzor wagi mieszania:
$w = (occ - start) / (end - start)$
Wyjasnienie symboli:
- $w$ to waga mieszania.
- $occ$ to wspolczynnik zajetosci.
- $start$ to prog startu mieszania.
- $end$ to prog pelnego przejscia do zlozonej nagrody.

Wzor nagrody mieszanej:
$reward = (1-w) * prosta + w * zlozona$
Wyjasnienie symboli:
- $reward$ to nagroda koncowa.
- $prosta$ to skladnik prostej nagrody.
- $zlozona$ to skladnik zlozonej nagrody.
- $w$ to waga mieszania.

## Przyspieszenia obliczen
1) Obliczenia sieci sa wykonywane na procesorze graficznym, co przyspiesza konwolucje i przetwarzanie paczek.
	Najwiekszy zysk jest widoczny, gdy wiele srodowisk dostarcza dane rownolegle.
2) Numba kompiluje wybrane funkcje w Pythonie do kodu maszynowego.
 	W praktyce przyspiesza funkcje przeszukiwania wszerz uzywane do oceny bezpieczenstwa, dzieki czemu krok srodowiska trwa krocej.
 	Chodzi o funkcje sprawdzajace, ile pol jest osiagalnych i czy istnieje bezpieczna sciezka od glowy do ogona.
 	To sa m.in. `_bfs_reachable_count` i `_bfs_path_exists`, ktore sa wywolywane przez `_flood_fill_ratio` i `_has_head_to_tail_path`.
	Jak to dziala technicznie: dekorator `@njit` uruchamia kompilacje just-in-time.
	Numba analizuje kod, inferuje typy zmiennych i generuje kod posredni (IR), a nastepnie kompiluje go do kodu maszynowego przez LLVM.
	Efekt jest taki, ze petle i operacje na tablicach NumPy sa wykonywane jako zoptymalizowany kod CPU zamiast wolnych operacji w interpreterze Pythona.
	Kompilacja odbywa sie przy pierwszym wywolaniu funkcji, a wynik jest cache'owany, wiec kolejne wywolania sa juz szybkie.
	W praktyce "ominiecie Pythona" oznacza, ze wewnatrz funkcji nie jest wykonywany bytecode Pythona.
	Interpreter uruchamia tylko wejsciowy "wrapper", a cale ciezkie petle dzialaja w kodzie maszynowym na surowych buforach pamieci tablic NumPy.
	Dzieki temu nie ma kosztu dynamicznego typowania, tworzenia obiektow Pythona i wolnych wywolan w kazdej iteracji petli.
	Pozostaje tylko jednorazowy narzut wywolania funkcji, a reszta jest wykonywana jak w kodzie C.
3) Cython kompiluje fragmenty bufora z priorytetami do rozszerzenia w C.
	W praktyce przyspiesza `sample_indices` (pobieranie indeksow lisci SumTree) oraz `batch_update` (zbiorcza aktualizacja priorytetow).
	Dzieki temu probkowanie paczki i aktualizacja priorytetow jest szybsza przy duzych buforach.
