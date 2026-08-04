"""Reviewed legal-reference corpus for the consumer workflow.

The corpus contains original editorial summaries and source metadata, not
copies of statutory text. Each provision occupies one synthetic page so a RAG
chunk can be mapped back to a precise authority without asking an LLM to invent
the citation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date
from functools import lru_cache

from app.consumer.schemas import (
    LegalAuthorityCitation,
    LegalProvision,
    LegalSource,
    ProvisionStatus,
)
from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument
from app.schemas.rag import Chunk, RetrievedChunk

CONSUMER_LAW_CORPUS_RELEASE_ID = "br-consumer-law-curated-2026-08-04-v1"
CORPUS_VERIFIED_ON = date(2026, 8, 4)
CONSTITUTION_URL = (
    "https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm"
)
CDC_URL = "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm"


def _provision(
    provision_id: str,
    source: LegalSource,
    article: str,
    summary: str,
    tags: Sequence[str],
    *,
    status: ProvisionStatus = ProvisionStatus.ACTIVE,
) -> LegalProvision:
    if source is LegalSource.FEDERAL_CONSTITUTION:
        source_name = "Constituição da República Federativa do Brasil de 1988"
        official_url = CONSTITUTION_URL
        citation_label = f"Constituição Federal, {article}"
    else:
        source_name = "Código de Defesa do Consumidor (Lei nº 8.078/1990)"
        official_url = CDC_URL
        citation_label = f"CDC, {article}"
    return LegalProvision(
        provision_id=provision_id,
        source=source,
        source_name=source_name,
        article=article,
        citation_label=citation_label,
        summary=summary,
        official_url=official_url,
        tags=list(tags),
        corpus_release_id=CONSUMER_LAW_CORPUS_RELEASE_ID,
        verified_on=CORPUS_VERIFIED_ON,
        status=status,
    )


_CF = LegalSource.FEDERAL_CONSTITUTION
_CDC = LegalSource.CONSUMER_DEFENSE_CODE

CURATED_PROVISIONS: tuple[LegalProvision, ...] = (
    _provision(
        "br-cf-art-1-iii",
        _CF,
        "art. 1º, III",
        "A dignidade da pessoa humana integra os fundamentos do Estado brasileiro e "
        "orienta a leitura das garantias aplicáveis ao caso.",
        ["dignidade", "fundamentos"],
    ),
    _provision(
        "br-cf-art-5-v",
        _CF,
        "art. 5º, V",
        "A Constituição assegura resposta proporcional à ofensa e possibilidade de "
        "reparação por prejuízos materiais, morais ou à imagem.",
        ["reparação", "dano material", "dano moral", "imagem"],
    ),
    _provision(
        "br-cf-art-5-x",
        _CF,
        "art. 5º, X",
        "Intimidade, vida privada, honra e imagem recebem proteção constitucional, com "
        "possibilidade de reparação quando houver violação.",
        ["privacidade", "honra", "imagem", "reparação"],
    ),
    _provision(
        "br-cf-art-5-xxxii",
        _CF,
        "art. 5º, XXXII",
        "A proteção do consumidor é um dever constitucional do Estado a ser concretizado "
        "por medidas legais e institucionais.",
        ["defesa do consumidor", "dever estatal"],
    ),
    _provision(
        "br-cf-art-5-xxxv",
        _CF,
        "art. 5º, XXXV",
        "Lesão ou ameaça a direito pode ser submetida ao Poder Judiciário; uma tentativa "
        "extrajudicial de acordo não elimina essa garantia.",
        ["acesso à justiça", "tutela judicial"],
    ),
    _provision(
        "br-cf-art-5-lxxix",
        _CF,
        "art. 5º, LXXIX",
        "A proteção de dados pessoais, também no ambiente digital, é direito fundamental "
        "e deve orientar o tratamento de informações do consumidor.",
        ["dados pessoais", "privacidade", "digital"],
    ),
    _provision(
        "br-cf-art-170-v",
        _CF,
        "art. 170, V",
        "A defesa do consumidor é princípio da ordem econômica e deve ser considerada no "
        "funcionamento das atividades empresariais.",
        ["ordem econômica", "defesa do consumidor"],
    ),
    _provision(
        "br-cdc-art-2",
        _CDC,
        "art. 2º",
        "Consumidor é quem adquire ou usa produto ou serviço como destinatário final; a "
        "coletividade que participa da relação também pode receber proteção.",
        ["conceito de consumidor", "destinatário final"],
    ),
    _provision(
        "br-cdc-art-3-p2",
        _CDC,
        "art. 3º, § 2º",
        "O conceito de serviço remunerado alcança atividades bancárias, financeiras, de "
        "crédito e de seguro, sem incluir relações trabalhistas.",
        ["banco", "serviço financeiro", "crédito", "seguro"],
    ),
    _provision(
        "br-cdc-art-4",
        _CDC,
        "art. 4º",
        "A política de consumo reconhece a vulnerabilidade do consumidor e busca relações "
        "transparentes, equilibradas e pautadas pela boa-fé.",
        ["vulnerabilidade", "boa-fé", "equilíbrio", "transparência"],
    ),
    _provision(
        "br-cdc-art-6",
        _CDC,
        "art. 6º",
        "Reúne direitos básicos como informação clara, proteção contra abusos, prevenção e "
        "reparação de danos, acesso a órgãos competentes e crédito responsável.",
        ["direitos básicos", "informação", "reparação", "crédito responsável"],
    ),
    _provision(
        "br-cdc-art-14",
        _CDC,
        "art. 14",
        "O prestador responde por danos ligados a defeito do serviço ou informação "
        "insuficiente, independentemente de culpa, ressalvadas as excludentes previstas.",
        ["responsabilidade", "defeito do serviço", "segurança"],
    ),
    _provision(
        "br-cdc-art-20",
        _CDC,
        "art. 20",
        "Diante de vício de qualidade do serviço, o consumidor pode escolher, conforme o "
        "caso, reexecução sem custo, restituição do pago ou abatimento do preço.",
        ["vício do serviço", "reexecução", "restituição", "abatimento"],
    ),
    _provision(
        "br-cdc-art-26",
        _CDC,
        "art. 26",
        "Reclamações por vícios aparentes seguem prazos de 30 dias para itens não duráveis e "
        "90 dias para duráveis, com regras próprias de início e impedimento da decadência.",
        ["prazo", "decadência", "vício"],
    ),
    _provision(
        "br-cdc-art-27",
        _CDC,
        "art. 27",
        "A pretensão de reparação por dano causado por fato de produto ou serviço tem prazo "
        "de cinco anos contado do conhecimento do dano e de sua autoria.",
        ["prazo", "prescrição", "reparação"],
    ),
    _provision(
        "br-cdc-art-30",
        _CDC,
        "art. 30",
        "Informação ou publicidade suficientemente precisa vincula o fornecedor que a "
        "divulga ou utiliza e passa a integrar o contrato celebrado.",
        ["oferta", "publicidade", "vinculação"],
    ),
    _provision(
        "br-cdc-art-35",
        _CDC,
        "art. 35",
        "Se a oferta não for cumprida, o consumidor pode escolher cumprimento, prestação "
        "equivalente ou encerramento do contrato com restituição e eventuais perdas e danos.",
        ["descumprimento de oferta", "restituição", "rescisão"],
    ),
    _provision(
        "br-cdc-art-42",
        _CDC,
        "art. 42",
        "A cobrança não pode constranger o consumidor. Valor indevido efetivamente pago pode "
        "gerar restituição em dobro, com atualização e juros, salvo engano justificável.",
        ["cobrança indevida", "pagamento", "restituição em dobro", "constrangimento"],
    ),
    _provision(
        "br-cdc-art-43",
        _CDC,
        "art. 43",
        "O consumidor pode acessar e corrigir cadastros. Registros devem ser claros e "
        "verdadeiros, e a abertura não solicitada e a correção seguem deveres de comunicação.",
        ["cadastro", "negativação", "correção", "dados"],
    ),
    _provision(
        "br-cdc-art-46",
        _CDC,
        "art. 46",
        "O contrato não obriga o consumidor quando não houve oportunidade prévia de conhecer "
        "seu conteúdo ou quando a redação dificulta compreender alcance e sentido.",
        ["contrato", "informação", "compreensão"],
    ),
    _provision(
        "br-cdc-art-47",
        _CDC,
        "art. 47",
        "Cláusulas contratuais de consumo devem ser interpretadas da maneira mais favorável "
        "ao consumidor.",
        ["contrato", "interpretação", "favorável ao consumidor"],
    ),
    _provision(
        "br-cdc-art-51",
        _CDC,
        "art. 51",
        "Cláusulas abusivas são nulas, inclusive quando criam desvantagem exagerada ou "
        "enfraquecem indevidamente a responsabilidade do fornecedor.",
        ["cláusula abusiva", "nulidade", "desvantagem exagerada"],
    ),
    _provision(
        "br-cdc-art-52",
        _CDC,
        "art. 52",
        "Crédito e financiamento exigem informação prévia sobre preço, juros, acréscimos, "
        "parcelas e total, além de redução proporcional na liquidação antecipada.",
        ["crédito", "juros", "custo", "liquidação antecipada"],
    ),
    _provision(
        "br-cdc-art-54-a",
        _CDC,
        "art. 54-A",
        "A disciplina do superendividamento protege a pessoa natural de boa-fé que não "
        "consegue pagar dívidas de consumo sem comprometer o mínimo existencial.",
        ["superendividamento", "boa-fé", "mínimo existencial"],
    ),
    _provision(
        "br-cdc-art-54-b",
        _CDC,
        "art. 54-B",
        "Oferta de crédito ou venda a prazo deve apresentar antecipadamente dados essenciais, "
        "incluindo custo efetivo total, juros, encargos, parcelas e total a pagar.",
        ["oferta de crédito", "custo efetivo total", "informação"],
    ),
    _provision(
        "br-cdc-art-54-c",
        _CDC,
        "art. 54-C",
        "Na oferta de crédito, são vedadas práticas que escondam riscos, dispensem avaliação "
        "responsável, pressionem pessoas vulneráveis ou condicionem negociação a renúncias.",
        ["oferta de crédito", "assédio", "vulnerabilidade", "risco"],
    ),
    _provision(
        "br-cdc-art-54-d",
        _CDC,
        "art. 54-D",
        "Antes da contratação, o fornecedor deve esclarecer custos e consequências, avaliar "
        "responsavelmente o crédito, identificar o financiador e fornecer cópia do contrato.",
        ["dever de informação", "avaliação de crédito", "contrato"],
    ),
    _provision(
        "br-cdc-art-54-e",
        _CDC,
        "art. 54-E",
        "O dispositivo foi integralmente vetado e, por isso, não oferece fundamento normativo "
        "autônomo para a notificação.",
        ["vetado", "superendividamento"],
        status=ProvisionStatus.VETOED,
    ),
    _provision(
        "br-cdc-art-54-f",
        _CDC,
        "art. 54-F",
        "Define situações em que contrato de crédito e compra financiada são conexos e prevê "
        "efeitos coordenados para arrependimento, inadimplemento ou invalidade.",
        ["contratos conexos", "crédito", "resolução"],
    ),
    _provision(
        "br-cdc-art-54-g",
        _CDC,
        "art. 54-G",
        "Em operações com crédito, limita cobranças de valor contestado no cartão quando "
        "observado o aviso prévio e reforça deveres de informação e entrega contratual.",
        ["cartão de crédito", "valor contestado", "fatura", "contrato"],
    ),
)


class LegalCorpus:
    """Immutable view over a versioned set of reviewed legal summaries."""

    def __init__(self, provisions: Sequence[LegalProvision]) -> None:
        if not provisions:
            raise ValueError("legal corpus cannot be empty")
        ids = [provision.provision_id for provision in provisions]
        if len(ids) != len(set(ids)):
            raise ValueError("legal corpus contains duplicate provision ids")
        releases = {provision.corpus_release_id for provision in provisions}
        if len(releases) != 1:
            raise ValueError("all provisions must belong to the same corpus release")
        self._provisions = tuple(provision.model_copy(deep=True) for provision in provisions)
        self._by_id = {provision.provision_id: provision for provision in self._provisions}

    @property
    def release_id(self) -> str:
        return self._provisions[0].corpus_release_id

    @property
    def provisions(self) -> tuple[LegalProvision, ...]:
        return self._provisions

    @property
    def active_provisions(self) -> tuple[LegalProvision, ...]:
        return tuple(
            provision
            for provision in self._provisions
            if provision.status is ProvisionStatus.ACTIVE
        )

    @property
    def corpus_sha256(self) -> str:
        manifest = "\n".join(
            f"{item.provision_id}:{item.content_sha256}" for item in self._provisions
        )
        return hashlib.sha256(manifest.encode("utf-8")).hexdigest()

    def get(self, provision_id: str) -> LegalProvision:
        try:
            return self._by_id[provision_id]
        except KeyError as exc:
            raise KeyError(f"unknown legal provision: {provision_id}") from exc

    def as_parsed_document(self) -> ParsedDocument:
        pages = [
            DocumentPage(number=index, text=self._page_text(provision))
            for index, provision in enumerate(self._provisions, start=1)
        ]
        return ParsedDocument(
            filename=f"{self.release_id}.txt",
            pages=pages,
            language="pt",
            extraction_method=ExtractionMethod.NATIVE_TEXT,
            warnings=[
                "Corpus de resumos editoriais; não contém transcrições oficiais.",
                "Confira a redação vigente no link oficial antes de uso jurídico.",
                "O art. 54-E do CDC é identificado como integralmente vetado.",
            ],
        )

    def provision_for_page(self, page: int) -> LegalProvision:
        if page < 1 or page > len(self._provisions):
            raise KeyError(f"page {page} is outside the legal corpus")
        return self._provisions[page - 1]

    def provisions_for_chunk(
        self, item: Chunk | RetrievedChunk
    ) -> tuple[LegalProvision, ...]:
        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        document = self.as_parsed_document()
        if chunk.doc_id != document.doc_id:
            raise ValueError("chunk does not belong to this legal corpus release")
        return tuple(
            self.provision_for_page(page)
            for page in range(chunk.page_start, chunk.page_end + 1)
        )

    def provision_for_chunk(self, item: Chunk | RetrievedChunk) -> LegalProvision:
        provisions = self.provisions_for_chunk(item)
        if len(provisions) != 1:
            raise ValueError("chunk spans more than one legal provision")
        return provisions[0]

    def authority_for_chunk(
        self,
        item: Chunk | RetrievedChunk,
        *,
        retrieval_rank: int | None = None,
    ) -> LegalAuthorityCitation:
        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        score = item.score if isinstance(item, RetrievedChunk) else None
        provision = self.provision_for_chunk(item)
        return LegalAuthorityCitation.from_provision(
            provision,
            chunk_id=chunk.chunk_id,
            retrieval_rank=retrieval_rank,
            retrieval_score=score,
        )

    @staticmethod
    def _page_text(provision: LegalProvision) -> str:
        status = "ATIVO" if provision.status is ProvisionStatus.ACTIVE else "VETADO"
        return (
            f"REFERÊNCIA LEGAL {provision.provision_id.upper()}\n\n"
            "Resumo editorial (não é transcrição oficial): "
            f"{provision.summary}\n\n"
            f"Citação: {provision.citation_label}\n"
            f"Status: {status}\n"
            f"Fonte oficial: {provision.official_url}\n"
            f"Versão do corpus: {provision.corpus_release_id}\n"
            f"SHA-256 do resumo: {provision.content_sha256}"
        )


@lru_cache(maxsize=1)
def get_default_legal_corpus() -> LegalCorpus:
    return LegalCorpus(CURATED_PROVISIONS)
