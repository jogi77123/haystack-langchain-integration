# haystack-langchain-integration
A program célja, hogy a haystack rendszerrel a helyi Library mappában, 
és almappáiban található fájlokat indexelje, és a Langchain program, és 
a faiss segítségével a langchain vectorstore mappájában létre hozza a 
index.faiss és index.pkl fájlokat. Az index fájlok, és a faiss dokumnetum 
tár segítségével a langchain programhoz kapcsolódó nyelvi modellek a kérdés 
válasz használata közben kereshetnek az adatbázisban, és a kérdésre generált 
válaszokat több forrásból is megerősíthetik, mielőtt a válasz létre jönne.
