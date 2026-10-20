
import json
import torch
import numpy as np
import faiss
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# ============================================================
# KONFIGURACJA
# ============================================================

INDEX_PATH = "regulamin.faiss"
METADATA_PATH = "fragmenty_regulaminu.json"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

MODEL_ID = "speakleash/Bielik-1.5B-v3.0-Instruct"


# ============================================================
# INICJALIZACJA MODELI I INDEKSU
# ============================================================

@st.cache_resource(show_spinner="Ładowanie modeli i indeksu FAISS...")
def init():
    # Tokenizer i model językowy
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    model.eval()

    # Model embeddingowy
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    # Indeks FAISS
    index = faiss.read_index(INDEX_PATH)

    # Metadane fragmentów
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Kontrola zgodności indeksu z metadanymi
    if index.ntotal != len(chunks):
        raise ValueError(
            f"Niezgodność danych: indeks FAISS zawiera "
            f"{index.ntotal} wektorów, a metadane "
            f"{len(chunks)} fragmentów."
        )

    return tokenizer, model, embedding_model, index, chunks


# ============================================================
# WYSZUKIWANIE KONTEKSTU
# ============================================================

def retrieve_context(
    question,
    embedding_model,
    index,
    chunks,
    top_k=4,
):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    # Nie pytamy FAISS o więcej wyników, niż ma indeks.
    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        chunk = chunks[int(idx)]

        results.append({
            "paragraf": chunk.get("paragraf"),
            "ustep": chunk.get("ustep"),
            "text": chunk["text"],
            "score": float(score),
        })

    context_parts = []

    for result in results:
        location = f"§ {result['paragraf']}"

        if result["ustep"] is not None:
            location += f", ust. {result['ustep']}"

        context_parts.append(
            f"Źródło: [{location}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    return context, results


# ============================================================
# GENEROWANIE ODPOWIEDZI
# ============================================================

def generate_answer(
    question,
    context,
    tokenizer,
    model,
    max_new_tokens=256,
):
    messages = [
        {
            "role": "system",
            "content": (
                """Jesteś asystentem odpowiadającym po polsku na pytania dotyczące regulaminu studiów.
                    Zasady odpowiedzi:
                    1. Odpowiadaj wyłącznie na podstawie informacji zawartych w dostarczonym kontekście regulaminu.
                    2. Każdą odpowiedź merytoryczną poprzyj wskazaniem źródła: numeru paragrafu (§) oraz ustępu, z którego pochodzi informacja.
                    3. Zachowuj dokładną numerację paragrafów i ustępów podaną w kontekście. Nie wymyślaj ani nie uzupełniaj numerów na podstawie własnej wiedzy.
                    4. Jeżeli odpowiedź wynika z kilku fragmentów, wskaż wszystkie odpowiednie paragrafy i ustępy.
                    5. Jeżeli kontekst nie zawiera wystarczających informacji, aby odpowiedzieć na pytanie, poinformuj: „Brak wystarczających informacji w dostarczonym kontekście regulaminu”.
                    6. Jeżeli fragment nie zawiera numeru paragrafu lub ustępu, nie zgaduj go. Wyraźnie zaznacz, że nie można ustalić pełnego źródła na podstawie dostarczonych danych.
                    7. Odpowiadaj konkretnie i zwięźle. Oddziel odpowiedź od wskazania źródła."""
                ),
        },
        {
            "role": "user",
            "content": (
                f"Kontekst regulaminu:\n{context}\n\n"
                f"Pytanie:\n{question}"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[
        0, inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# INTERFEJS STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Asystent regulaminu studiów",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Asystent regulaminu studiów")

st.caption(
    "Zadawaj pytania dotyczące regulaminu. "
    "Odpowiedzi są generowane na podstawie wyszukanych fragmentów."
)

with st.sidebar:
    st.header("Ustawienia")

    top_k = st.slider(
        "Liczba wyszukiwanych fragmentów",
        min_value=1,
        max_value=10,
        value=4,
    )

    max_new_tokens = st.slider(
        "Maksymalna długość odpowiedzi",
        min_value=64,
        max_value=512,
        value=256,
        step=64,
    )

    show_sources = st.checkbox(
        "Pokaż źródła i wyniki wyszukiwania",
        value=True,
    )

    if st.button("Wyczyść historię", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Załadowanie modeli tylko raz
try:
    tokenizer, model, embedding_model, index, chunks = init()
except Exception as exc:
    st.error(f"Nie udało się zainicjalizować aplikacji: {exc}")
    st.stop()


# Informacje o działaniu
with st.sidebar:
    st.divider()
    st.caption(f"Model językowy: {MODEL_ID}")
    st.caption(f"Urządzenie: {model.device}")
    st.caption(f"Fragmenty w bazie: {len(chunks)}")


# Historia czatu
if "messages" not in st.session_state:
    st.session_state.messages = []


# Odtworzenie historii
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if show_sources and message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("Źródła odpowiedzi"):
                    for i, source in enumerate(sources, start=1):
                        paragraf = source["paragraf"]
                        ustep = source["ustep"]

                        location = f"§ {paragraf}"
                        if ustep is not None:
                            location += f", ust. {ustep}"

                        st.markdown(f"**Fragment {i}: {location}**")
                        st.caption(
                            f"Wynik wyszukiwania: "
                            f"{source['score']:.4f}"
                        )
                        st.write(source["text"])
                        st.divider()


# Pole pytania
question = st.chat_input(
    "Zadaj pytanie dotyczące regulaminu studiów..."
)

if question:
    # Wyświetlenie i zapis pytania
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Wyszukiwanie fragmentów regulaminu..."):
                context, results = retrieve_context(
                    question=question,
                    embedding_model=embedding_model,
                    index=index,
                    chunks=chunks,
                    top_k=top_k,
                )

            if not results:
                answer = (
                    "Brak wystarczających informacji "
                    "w dostarczonym kontekście regulaminu."
                )
            else:
                with st.spinner("Generowanie odpowiedzi przez Bielika..."):
                    answer = generate_answer(
                        question=question,
                        context=context,
                        tokenizer=tokenizer,
                        model=model,
                        max_new_tokens=max_new_tokens,
                    )

            st.markdown(answer)

            if show_sources and results:
                with st.expander("Źródła odpowiedzi", expanded=True):
                    for i, result in enumerate(results, start=1):
                        location = f"§ {result['paragraf']}"

                        if result["ustep"] is not None:
                            location += f", ust. {result['ustep']}"

                        st.markdown(f"**Fragment {i}: {location}**")
                        st.caption(
                            f"Wynik wyszukiwania: {result['score']:.4f}"
                        )
                        st.write(result["text"])
                        st.divider()

            # Zachowanie odpowiedzi wraz ze źródłami
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": results,
            })

        except Exception as exc:
            st.error(f"Wystąpił błąd podczas generowania odpowiedzi: {exc}")
