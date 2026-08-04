"""Versioned legal corpus for the consumer workflow.

The CDC portion is built from an integrity-checked offline snapshot of the
complete compiled statute published by Planalto.  The Constitution portion is
a small reviewed set of consumer-relevant provisions transcribed from the
official compiled Constitution.  Editorial summaries remain visibly separate
from the official text in every page and citation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date
from functools import lru_cache
from types import MappingProxyType

from app.consumer.cdc_snapshot import (
    CDC_PARSER_VERSION,
    ParsedCdcArticle,
    load_manifest,
    load_official_cdc,
)
from app.consumer.schemas import (
    LegalAuthorityCitation,
    LegalProvision,
    LegalSource,
    LegalTextUnit,
    ProvisionStatus,
)
from app.schemas.document import DocumentPage, ExtractionMethod, ParsedDocument
from app.schemas.rag import Chunk, MetadataValue, RetrievedChunk

CONSUMER_LAW_CORPUS_RELEASE_ID = "br-consumer-law-2026-08-04-v3"
CORPUS_VERIFIED_ON = date(2026, 8, 4)
CONSTITUTION_URL = "https://www.planalto.gov.br/ccivil_03/constituicao/constituicaocompilado.htm"
CDC_URL = "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm"

_SOURCE_NAME_CF = "Constituição da República Federativa do Brasil de 1988"
_SOURCE_NAME_CDC = "Código de Defesa do Consumidor (Lei nº 8.078/1990)"
_LEGACY_ID_ALIASES = {"br-cdc-art-3-p2": "br-cdc-art-3"}
LEGAL_CHUNKING_VERSION = "legal-hierarchy-v1"


def _cf_provision(
    provision_id: str,
    article: str,
    summary: str,
    tags: Sequence[str],
    official_text: str,
) -> LegalProvision:
    return LegalProvision(
        provision_id=provision_id,
        source=LegalSource.FEDERAL_CONSTITUTION,
        source_name=_SOURCE_NAME_CF,
        article=article,
        citation_label=f"Constituição Federal, {article}",
        summary=summary,
        official_url=CONSTITUTION_URL,
        tags=tuple(tags),
        corpus_release_id=CONSUMER_LAW_CORPUS_RELEASE_ID,
        verified_on=CORPUS_VERIFIED_ON,
        law_id="br-cf",
        official_text=official_text,
    )


_CONSTITUTION_PROVISIONS: tuple[LegalProvision, ...] = (
    _cf_provision(
        "br-cf-art-1-iii",
        "art. 1º, III",
        "A dignidade da pessoa humana integra os fundamentos do Estado brasileiro e "
        "orienta a leitura das garantias aplicáveis ao caso.",
        ["dignidade", "fundamentos"],
        "III - a dignidade da pessoa humana;",
    ),
    _cf_provision(
        "br-cf-art-5-v",
        "art. 5º, V",
        "A Constituição assegura resposta proporcional à ofensa e possibilidade de "
        "reparação por prejuízos materiais, morais ou à imagem.",
        ["reparação", "dano material", "dano moral", "imagem"],
        "V - é assegurado o direito de resposta, proporcional ao agravo, além da "
        "indenização por dano material, moral ou à imagem;",
    ),
    _cf_provision(
        "br-cf-art-5-x",
        "art. 5º, X",
        "Intimidade, vida privada, honra e imagem recebem proteção constitucional, com "
        "possibilidade de reparação quando houver violação.",
        ["privacidade", "honra", "imagem", "reparação"],
        "X - são invioláveis a intimidade, a vida privada, a honra e a imagem das "
        "pessoas, assegurado o direito a indenização pelo dano material ou moral "
        "decorrente de sua violação;",
    ),
    _cf_provision(
        "br-cf-art-5-xxxii",
        "art. 5º, XXXII",
        "A proteção do consumidor é um dever constitucional do Estado a ser concretizado "
        "por medidas legais e institucionais.",
        ["defesa do consumidor", "dever estatal"],
        "XXXII - o Estado promoverá, na forma da lei, a defesa do consumidor;",
    ),
    _cf_provision(
        "br-cf-art-5-xxxv",
        "art. 5º, XXXV",
        "Lesão ou ameaça a direito pode ser submetida ao Poder Judiciário; uma tentativa "
        "extrajudicial de acordo não elimina essa garantia.",
        ["acesso à justiça", "tutela judicial"],
        "XXXV - a lei não excluirá da apreciação do Poder Judiciário lesão ou ameaça a direito;",
    ),
    _cf_provision(
        "br-cf-art-5-lxxix",
        "art. 5º, LXXIX",
        "A proteção de dados pessoais, também no ambiente digital, é direito fundamental "
        "e deve orientar o tratamento de informações do consumidor.",
        ["dados pessoais", "privacidade", "digital"],
        "LXXIX - é assegurado, nos termos da lei, o direito à proteção dos dados "
        "pessoais, inclusive nos meios digitais.",
    ),
    _cf_provision(
        "br-cf-art-170-v",
        "art. 170, V",
        "A defesa do consumidor é princípio da ordem econômica e deve ser considerada no "
        "funcionamento das atividades empresariais.",
        ["ordem econômica", "defesa do consumidor"],
        "V - defesa do consumidor;",
    ),
)


# Human-reviewed summaries remain useful for notice generation.  Retrieval is
# performed over official text, so articles not listed here still have complete
# coverage and receive a deterministic caput extract as their display summary.
_REVIEWED_CDC_METADATA: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "2": (
        "Consumidor é quem adquire ou usa produto ou serviço como destinatário final; a "
        "coletividade que participa da relação também pode receber proteção.",
        ("conceito de consumidor", "destinatário final"),
    ),
    "3": (
        "Define fornecedor, produto e serviço; atividades bancárias, financeiras, de "
        "crédito e de seguro integram o conceito de serviço remunerado.",
        ("fornecedor", "produto", "serviço", "banco", "crédito", "seguro"),
    ),
    "4": (
        "A política de consumo reconhece a vulnerabilidade do consumidor e busca relações "
        "transparentes, equilibradas e pautadas pela boa-fé.",
        ("vulnerabilidade", "boa-fé", "equilíbrio", "transparência"),
    ),
    "6": (
        "Reúne direitos básicos como informação clara, proteção contra abusos, prevenção "
        "e reparação de danos, acesso a órgãos competentes e crédito responsável.",
        ("direitos básicos", "informação", "reparação", "crédito responsável"),
    ),
    "14": (
        "O prestador responde por danos ligados a defeito do serviço ou informação "
        "insuficiente, independentemente de culpa, ressalvadas as excludentes previstas.",
        ("responsabilidade", "defeito do serviço", "segurança"),
    ),
    "18": (
        "Disciplina a responsabilidade solidária por vícios de produtos e as alternativas "
        "de substituição, restituição ou abatimento disponíveis ao consumidor.",
        ("vício do produto", "substituição", "restituição", "abatimento"),
    ),
    "20": (
        "Diante de vício de qualidade do serviço, o consumidor pode escolher, conforme o "
        "caso, reexecução sem custo, restituição do pago ou abatimento do preço.",
        ("vício do serviço", "reexecução", "restituição", "abatimento"),
    ),
    "22": (
        "Órgãos públicos e concessionárias devem fornecer serviços adequados, eficientes, "
        "seguros e, quando essenciais, contínuos.",
        ("serviço público", "continuidade", "eficiência", "segurança"),
    ),
    "26": (
        "Reclamações por vícios aparentes seguem prazos de 30 dias para itens não duráveis "
        "e 90 dias para duráveis, com regras próprias de início e impedimento.",
        ("prazo", "decadência", "vício"),
    ),
    "27": (
        "A pretensão de reparação por dano causado por fato de produto ou serviço tem prazo "
        "de cinco anos contado do conhecimento do dano e de sua autoria.",
        ("prazo", "prescrição", "reparação"),
    ),
    "30": (
        "Informação ou publicidade suficientemente precisa vincula o fornecedor que a "
        "divulga ou utiliza e passa a integrar o contrato celebrado.",
        ("oferta", "publicidade", "vinculação"),
    ),
    "35": (
        "Se a oferta não for cumprida, o consumidor pode escolher cumprimento, prestação "
        "equivalente ou encerramento do contrato com restituição e perdas e danos.",
        ("descumprimento de oferta", "restituição", "rescisão"),
    ),
    "39": (
        "Proíbe práticas abusivas no fornecimento de produtos e serviços, inclusive venda "
        "casada, vantagem excessiva e fornecimento não solicitado.",
        ("prática abusiva", "venda casada", "vantagem excessiva"),
    ),
    "42": (
        "A cobrança não pode constranger o consumidor. Valor indevido efetivamente pago "
        "pode gerar restituição em dobro, salvo engano justificável.",
        ("cobrança indevida", "restituição em dobro", "constrangimento"),
    ),
    "43": (
        "O consumidor pode acessar e corrigir cadastros; registros devem ser claros e "
        "verdadeiros e sua abertura ou correção segue deveres de comunicação.",
        ("cadastro", "negativação", "correção", "dados"),
    ),
    "46": (
        "O contrato não obriga o consumidor quando não houve oportunidade prévia de "
        "conhecer seu conteúdo ou a redação impede sua compreensão.",
        ("contrato", "informação", "compreensão"),
    ),
    "47": (
        "Cláusulas contratuais de consumo devem ser interpretadas da maneira mais "
        "favorável ao consumidor.",
        ("contrato", "interpretação", "favorável ao consumidor"),
    ),
    "49": (
        "Permite desistir, em sete dias, de contratação realizada fora do estabelecimento, "
        "com devolução imediata e atualizada dos valores pagos.",
        ("direito de arrependimento", "contratação à distância", "devolução"),
    ),
    "51": (
        "Cláusulas abusivas são nulas, inclusive quando criam desvantagem exagerada ou "
        "enfraquecem indevidamente a responsabilidade do fornecedor.",
        ("cláusula abusiva", "nulidade", "desvantagem exagerada"),
    ),
    "52": (
        "Crédito e financiamento exigem informação prévia sobre preço, juros, acréscimos, "
        "parcelas e total, além de redução proporcional na liquidação antecipada.",
        ("crédito", "juros", "custo", "liquidação antecipada"),
    ),
    "54-a": (
        "A disciplina do superendividamento protege a pessoa natural de boa-fé que não "
        "consegue pagar dívidas de consumo sem comprometer o mínimo existencial.",
        ("superendividamento", "boa-fé", "mínimo existencial"),
    ),
    "54-b": (
        "Oferta de crédito ou venda a prazo deve apresentar antecipadamente custo efetivo "
        "total, juros, encargos, parcelas e total a pagar.",
        ("oferta de crédito", "custo efetivo total", "informação"),
    ),
    "54-c": (
        "Na oferta de crédito, são vedadas práticas que escondam riscos, dispensem avaliação "
        "responsável ou pressionem pessoas vulneráveis.",
        ("oferta de crédito", "assédio", "vulnerabilidade", "risco"),
    ),
    "54-d": (
        "Antes da contratação, o fornecedor deve esclarecer custos e consequências, avaliar "
        "responsavelmente o crédito e fornecer cópia do contrato.",
        ("dever de informação", "avaliação de crédito", "contrato"),
    ),
    "54-e": (
        "O dispositivo foi integralmente vetado e não oferece fundamento normativo "
        "autônomo para a notificação.",
        ("vetado", "superendividamento"),
    ),
    "54-f": (
        "Define situações em que contrato de crédito e compra financiada são conexos e "
        "prevê efeitos coordenados para arrependimento, inadimplemento ou invalidade.",
        ("contratos conexos", "crédito", "resolução"),
    ),
    "54-g": (
        "Em operações com crédito, limita cobranças de valor contestado no cartão e reforça "
        "deveres de informação e entrega contratual.",
        ("cartão de crédito", "valor contestado", "fatura", "contrato"),
    ),
    "104-a": (
        "Prevê audiência conciliatória global para repactuação das dívidas do consumidor "
        "superendividado, preservando o mínimo existencial.",
        ("superendividamento", "conciliação", "plano de pagamento"),
    ),
    "104-b": (
        "Disciplina o processo judicial de revisão e repactuação compulsória quando a "
        "conciliação com credores não for integralmente bem-sucedida.",
        ("superendividamento", "repactuação judicial", "credores"),
    ),
    "104-c": (
        "Autoriza órgãos públicos de defesa do consumidor a conduzir preventivamente a "
        "conciliação administrativa por superendividamento.",
        ("superendividamento", "conciliação administrativa", "órgãos públicos"),
    ),
}


def _cdc_provisions() -> tuple[LegalProvision, ...]:
    manifest, articles = load_official_cdc()
    provisions: list[LegalProvision] = []
    for article in articles:
        reviewed = _REVIEWED_CDC_METADATA.get(article.article_key)
        summary, tags = reviewed or (_caput_extract(article), ())
        provisions.append(
            LegalProvision(
                provision_id=article.provision_id,
                source=LegalSource.CONSUMER_DEFENSE_CODE,
                source_name=_SOURCE_NAME_CDC,
                article=article.article_label,
                citation_label=f"CDC, {article.article_label}",
                summary=summary,
                official_url=CDC_URL,
                tags=tuple(tags),
                corpus_release_id=CONSUMER_LAW_CORPUS_RELEASE_ID,
                verified_on=manifest.retrieved_on,
                status=article.status,
                law_id="br-cdc",
                article_key=article.article_key,
                title=article.title,
                chapter=article.chapter,
                section=article.section,
                official_text=article.official_text,
                source_snapshot_sha256=manifest.snapshot_sha256,
                units=tuple(article.units),
            )
        )
    return tuple(provisions)


def _caput_extract(article: ParsedCdcArticle) -> str:
    caput = article.units[0].text
    caput = re.sub(r"^Art\.\s*\d+(?:[º°])?(?:-[A-Z])?\.?(?:\s+|$)", "", caput)
    if len(caput) <= 700:
        return caput
    sentence_end = caput.rfind(". ", 0, 700)
    return caput[: sentence_end + 1 if sentence_end >= 200 else 697].rstrip() + "..."


CURATED_PROVISIONS: tuple[LegalProvision, ...] = (
    *_CONSTITUTION_PROVISIONS,
    *_cdc_provisions(),
)


class LegalCorpus:
    """Immutable view over versioned legal authorities and their official text."""

    def __init__(self, provisions: Sequence[LegalProvision]) -> None:
        if not provisions:
            raise ValueError("legal corpus cannot be empty")
        ids = [provision.provision_id for provision in provisions]
        if len(ids) != len(set(ids)):
            raise ValueError("legal corpus contains duplicate provision ids")
        releases = {provision.corpus_release_id for provision in provisions}
        if len(releases) != 1:
            raise ValueError("all provisions must belong to the same corpus release")
        self._provisions = tuple(
            LegalProvision.model_validate(provision.model_dump(mode="python"))
            for provision in provisions
        )
        self._validate_cdc_snapshot_provenance()
        self._by_id = MappingProxyType(
            {provision.provision_id: provision for provision in self._provisions}
        )
        self._corpus_sha256 = self._calculate_corpus_sha256()

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
        return self._corpus_sha256

    def _validate_cdc_snapshot_provenance(self) -> None:
        cdc_provisions = tuple(
            provision
            for provision in self._provisions
            if provision.source is LegalSource.CONSUMER_DEFENSE_CODE
        )
        if not cdc_provisions:
            return
        manifest = load_manifest()
        for provision in cdc_provisions:
            if provision.source_snapshot_sha256 != manifest.snapshot_sha256:
                raise ValueError(f"{provision.provision_id} does not match the pinned CDC snapshot")
            if provision.official_text is None or provision.official_text_sha256 is None:
                raise ValueError(f"{provision.provision_id} has no integrity-checked official text")

    def _calculate_corpus_sha256(self) -> str:
        cdc_manifest = (
            load_manifest()
            if any(
                provision.source is LegalSource.CONSUMER_DEFENSE_CODE
                for provision in self._provisions
            )
            else None
        )
        source_manifests = []
        if cdc_manifest is not None:
            source_manifests.append(
                {
                    "law_id": cdc_manifest.law_id,
                    "manifest_schema_version": cdc_manifest.schema_version,
                    "manifest_release_id": cdc_manifest.release_id,
                    "source_url": cdc_manifest.source_url,
                    "retrieved_on": cdc_manifest.retrieved_on.isoformat(),
                    "encoding": cdc_manifest.encoding,
                    "snapshot_file": cdc_manifest.snapshot_file,
                    "snapshot_sha256": cdc_manifest.snapshot_sha256,
                    "manifest_parser_version": cdc_manifest.parser_version,
                    "runtime_parser_version": CDC_PARSER_VERSION,
                    "acquisition_method": cdc_manifest.acquisition_method,
                    "acquisition_note": cdc_manifest.acquisition_note,
                    "final_url": cdc_manifest.final_url,
                    "http_etag": cdc_manifest.http_etag,
                    "http_last_modified": cdc_manifest.http_last_modified,
                    "refresh_tool_version": cdc_manifest.refresh_tool_version,
                    "review_status": cdc_manifest.review_status,
                }
            )
        payload = {
            "schema_version": 1,
            "corpus_release_id": self.release_id,
            "source_manifests": source_manifests,
            "provisions": [
                {
                    "position": position,
                    "provision_id": provision.provision_id,
                    "law_id": provision.law_id,
                    "source": provision.source.value,
                    "source_name": provision.source_name,
                    "article": provision.article,
                    "article_key": provision.article_key,
                    "citation_label": provision.citation_label,
                    "official_url": provision.official_url,
                    "verified_on": provision.verified_on.isoformat(),
                    "status": provision.status.value,
                    "hierarchy": {
                        "title": provision.title,
                        "chapter": provision.chapter,
                        "section": provision.section,
                    },
                    "summary": provision.summary,
                    "summary_sha256": provision.content_sha256,
                    "tags": sorted(set(provision.tags)),
                    "official_text_sha256": provision.official_text_sha256,
                    "source_snapshot_sha256": provision.source_snapshot_sha256,
                    "units": [
                        {
                            "position": unit_position,
                            "unit_id": unit.unit_id,
                            "kind": unit.kind.value,
                            "label": unit.label,
                            "status": unit.status.value,
                            "hierarchy": {
                                "paragraph": unit.paragraph,
                                "inciso": unit.inciso,
                                "alinea": unit.alinea,
                            },
                            "content_sha256": unit.content_sha256,
                        }
                        for unit_position, unit in enumerate(provision.units, start=1)
                    ],
                }
                for position, provision in enumerate(self._provisions, start=1)
            ],
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def get(self, provision_id: str) -> LegalProvision:
        canonical_id = _LEGACY_ID_ALIASES.get(provision_id, provision_id)
        try:
            return self._by_id[canonical_id]
        except KeyError as exc:
            raise KeyError(f"unknown legal provision: {provision_id}") from exc

    def as_parsed_document(self) -> ParsedDocument:
        """Return one synthetic page per article, safe for the generic chunker."""

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
                "CDC: texto oficial compilado de snapshot local verificado por SHA-256.",
                "CF: seleção de dispositivos transcritos da compilação oficial do Planalto.",
                "Unidades vetadas ou revogadas são mantidas para auditoria e marcadas.",
            ],
        )

    def as_chunks(
        self,
        *,
        target_chars: int = 1_200,
        include_inactive: bool = False,
    ) -> list[Chunk]:
        """Create hierarchy-aware chunks that never cross article boundaries."""

        if target_chars < 200:
            raise ValueError("target_chars must be at least 200")
        document = self.as_parsed_document()
        chunks: list[Chunk] = []
        for page_number, provision in enumerate(self._provisions, start=1):
            units: Sequence[LegalTextUnit | None] = provision.units or (None,)
            for unit in units:
                if (
                    unit is not None
                    and not include_inactive
                    and unit.status is not ProvisionStatus.ACTIVE
                ):
                    continue
                body = provision.official_text or provision.summary if unit is None else unit.text
                context = self._chunk_context(provision, unit)
                for piece_number, piece in enumerate(
                    _split_text(body, target_chars - len(context) - 2), start=1
                ):
                    unit_key = provision.provision_id if unit is None else unit.unit_id
                    chunk_id = f"{document.doc_id}:legal:{unit_key}:part-{piece_number:02d}"
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=document.doc_id,
                            text=f"{context}\n\n{piece}",
                            section=self._section_label(provision),
                            page_start=page_number,
                            page_end=page_number,
                            metadata=self._legal_metadata(
                                provision,
                                unit,
                                page=page_number,
                                chunking_version=(
                                    f"{LEGAL_CHUNKING_VERSION}:target={target_chars}"
                                ),
                            ),
                        )
                    )
        return chunks

    def provision_for_page(self, page: int) -> LegalProvision:
        if page < 1 or page > len(self._provisions):
            raise KeyError(f"page {page} is outside the legal corpus")
        return self._provisions[page - 1]

    def provisions_for_chunk(self, item: Chunk | RetrievedChunk) -> tuple[LegalProvision, ...]:
        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        document = self.as_parsed_document()
        if chunk.doc_id != document.doc_id:
            raise ValueError("chunk does not belong to this legal corpus release")
        return tuple(
            self.provision_for_page(page) for page in range(chunk.page_start, chunk.page_end + 1)
        )

    def provision_for_chunk(self, item: Chunk | RetrievedChunk) -> LegalProvision:
        provisions = self.provisions_for_chunk(item)
        if len(provisions) != 1:
            raise ValueError("chunk spans more than one legal provision")
        return provisions[0]

    def unit_for_chunk(self, item: Chunk | RetrievedChunk) -> LegalTextUnit | None:
        """Return the normative unit for an ``as_chunks`` result, when present."""

        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        provision = self.provision_for_chunk(chunk)
        for unit in provision.units:
            if f":legal:{unit.unit_id}:part-" in chunk.chunk_id:
                return unit
        return None

    def metadata_for_chunk(self, item: Chunk | RetrievedChunk) -> dict[str, MetadataValue]:
        """Return embedded metadata or reconstruct it for legacy chunks."""

        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        if chunk.metadata:
            return dict(chunk.metadata)
        provision = self.provision_for_chunk(chunk)
        unit = self.unit_for_chunk(chunk)
        return self._legal_metadata(provision, unit, page=chunk.page_start)

    @staticmethod
    def _legal_metadata(
        provision: LegalProvision,
        unit: LegalTextUnit | None,
        *,
        page: int,
        chunking_version: str = LEGAL_CHUNKING_VERSION,
    ) -> dict[str, MetadataValue]:
        content_kind = "official" if provision.official_text is not None or unit else "editorial"
        return {
            "law_id": provision.law_id,
            "provision_id": provision.provision_id,
            "article": provision.article,
            "article_key": provision.article_key,
            "title": provision.title,
            "chapter": provision.chapter,
            "section": provision.section,
            "unit_id": unit.unit_id if unit else None,
            "unit_kind": unit.kind.value if unit else None,
            "paragraph": unit.paragraph if unit else None,
            "inciso": unit.inciso if unit else None,
            "alinea": unit.alinea if unit else None,
            "status": (unit.status if unit else provision.status).value,
            "content_kind": content_kind,
            "chunking_version": chunking_version,
            "official_url": provision.official_url,
            "corpus_release_id": provision.corpus_release_id,
            "verified_on": provision.verified_on.isoformat(),
            "content_sha256": (
                unit.content_sha256
                if unit is not None
                else provision.official_text_sha256 or provision.content_sha256
            ),
            "source_snapshot_sha256": provision.source_snapshot_sha256,
            "page": page,
        }

    def authority_for_chunk(
        self,
        item: Chunk | RetrievedChunk,
        *,
        retrieval_rank: int | None = None,
    ) -> LegalAuthorityCitation:
        chunk = item.chunk if isinstance(item, RetrievedChunk) else item
        score = item.score if isinstance(item, RetrievedChunk) else None
        provision = self.provision_for_chunk(item)
        unit = self.unit_for_chunk(item)
        return LegalAuthorityCitation.from_provision(
            provision,
            unit=unit,
            chunk_id=chunk.chunk_id,
            retrieval_rank=retrieval_rank,
            retrieval_score=score,
        )

    @staticmethod
    def _page_text(provision: LegalProvision) -> str:
        status = provision.status.value.upper()
        if provision.official_text is None:
            content_label = "Resumo editorial (não é transcrição oficial)"
            content = provision.summary
        else:
            content_label = "Texto oficial compilado"
            active_units = [
                unit.text for unit in provision.units if unit.status is ProvisionStatus.ACTIVE
            ]
            content = "\n\n".join(active_units) or provision.official_text
        hierarchy = " > ".join(
            item for item in (provision.title, provision.chapter, provision.section) if item
        )
        return (
            f"REFERÊNCIA LEGAL {provision.provision_id.upper()}\n\n"
            f"Citação: {provision.citation_label}\n"
            f"Hierarquia: {hierarchy or 'não informada'}\n"
            f"Status: {status}\n"
            f"Fonte oficial: {provision.official_url}\n"
            f"Versão do corpus: {provision.corpus_release_id}\n"
            f"SHA-256 do conteúdo: "
            f"{provision.official_text_sha256 or provision.content_sha256}\n\n"
            f"{content_label}:\n{content}"
        )

    @staticmethod
    def _section_label(provision: LegalProvision) -> str:
        return " > ".join(
            item
            for item in (
                provision.source_name,
                provision.title,
                provision.chapter,
                provision.section,
                provision.article,
            )
            if item
        )

    @classmethod
    def _chunk_context(cls, provision: LegalProvision, unit: LegalTextUnit | None) -> str:
        fields = [
            cls._section_label(provision),
            f"Citação: {provision.citation_label}",
            f"Status: {(unit.status if unit else provision.status).value}",
            "Conteúdo: "
            + (
                "texto oficial"
                if provision.official_text is not None or unit is not None
                else "resumo editorial (não é transcrição oficial)"
            ),
            f"Fonte: {provision.official_url}",
        ]
        if unit is not None:
            fields.insert(1, f"Unidade: {unit.label} ({unit.unit_id})")
        return "\n".join(fields)


def _split_text(text: str, max_chars: int) -> list[str]:
    if max_chars < 100:
        raise ValueError("target_chars is too small for legal provenance context")
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        boundary = max(
            remaining.rfind(". ", 0, max_chars),
            remaining.rfind("; ", 0, max_chars),
            remaining.rfind(" ", 0, max_chars),
        )
        if boundary < max_chars // 2:
            boundary = max_chars
        else:
            boundary += 1
        pieces.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


@lru_cache(maxsize=1)
def get_default_legal_corpus() -> LegalCorpus:
    return LegalCorpus(CURATED_PROVISIONS)
