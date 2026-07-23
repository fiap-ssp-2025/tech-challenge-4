"""Transcrição completa de arquivos de áudio usando Azure AI Speech."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class AzureSTTError(RuntimeError):
    """Erro ao transcrever áudio usando Azure Speech."""


def transcrever_audio(
    path: str | Path,
    *,
    idioma: str = "pt-BR",
    timeout: float = 120.0,
) -> str:
    """Transcreve todo o arquivo de áudio usando reconhecimento contínuo."""

    audio_path = Path(path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    speech_key = os.getenv("AZURE_SPEECH_KEY", "").strip()
    speech_region = os.getenv("AZURE_SPEECH_REGION", "").strip()

    if not speech_key:
        raise AzureSTTError("AZURE_SPEECH_KEY não configurada no arquivo .env.")

    if not speech_region:
        raise AzureSTTError("AZURE_SPEECH_REGION não configurada no arquivo .env.")

    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region,
        )
        speech_config.speech_recognition_language = idioma

        audio_config = speechsdk.audio.AudioConfig(
            filename=str(audio_path),
        )

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        transcricoes: list[str] = []
        erros: list[str] = []
        terminou = threading.Event()

        def ao_reconhecer(
            evt: speechsdk.SpeechRecognitionEventArgs,
        ) -> None:
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                texto = evt.result.text.strip()

                if texto:
                    transcricoes.append(texto)
                    print(f"[Azure] Trecho reconhecido: {texto}")

        def ao_cancelar(
            evt: speechsdk.SpeechRecognitionCanceledEventArgs,
        ) -> None:
            cancelamento = evt.result.cancellation_details

            # EndOfStream é o encerramento normal de um arquivo de áudio.
            if cancelamento.reason == speechsdk.CancellationReason.EndOfStream:
                terminou.set()
                return

            erros.append(
                "Reconhecimento cancelado pelo Azure Speech. "
                f"Motivo: {cancelamento.reason}. "
                f"Detalhes: "
                f"{cancelamento.error_details or 'não informados'}"
            )
            terminou.set()

        def ao_encerrar_sessao(
            _evt: speechsdk.SessionEventArgs,
        ) -> None:
            terminou.set()

        recognizer.recognized.connect(ao_reconhecer)
        recognizer.canceled.connect(ao_cancelar)
        recognizer.session_stopped.connect(ao_encerrar_sessao)

        recognizer.start_continuous_recognition_async().get()

        finalizou = terminou.wait(timeout=timeout)

        recognizer.stop_continuous_recognition_async().get()

    except AzureSTTError:
        raise

    except Exception as exc:
        raise AzureSTTError(
            f"Falha ao executar o Azure Speech: {exc}"
        ) from exc

    if not finalizou:
        raise AzureSTTError(
            f"O reconhecimento ultrapassou o limite de {timeout:.0f} segundos."
        )

    if erros:
        raise AzureSTTError(erros[0])

    texto_final = " ".join(transcricoes).strip()

    if not texto_final:
        raise AzureSTTError(
            "O Azure Speech terminou o áudio sem reconhecer nenhuma fala."
        )

    return texto_final


if __name__ == "__main__":
    caminho = input("Arquivo de áudio: ").strip()

    try:
        resultado = transcrever_audio(caminho)

        print("\n===== TRANSCRIÇÃO COMPLETA =====\n")
        print(resultado)

    except (FileNotFoundError, AzureSTTError) as exc:
        print(f"\nErro: {exc}")