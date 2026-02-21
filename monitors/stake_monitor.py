"""
Monitor de Stake.com para detectar Big Wins y eventos de casino.

Utiliza GraphQL API para:
- Detectar victorias grandes (Big Wins)
- Monitorear High Rollers de deportes
- Capturar datos de apuestas en tiempo real

NOTA IMPORTANTE:
Stake.com tiene una API GraphQL con queries whitelisted.
Las queries deben coincidir EXACTAMENTE con las de su frontend.
Para monitoreo de casino, se recomienda usar el KickMonitor
para detectar momentos virales en streams de jugadores de Stake.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import httpx

from .base_monitor import BaseMonitor, MonitorEvent, EventType

# Intentar importar curl_cffi para bypass de Cloudflare
try:
    from curl_cffi.requests import AsyncSession
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False


@dataclass
class StakeBet:
    """Estructura de datos para una apuesta de Stake."""
    bet_id: str
    game: str
    user: str
    amount: float
    payout: float
    multiplier: float
    currency: str
    timestamp: datetime
    is_big_win: bool = False

    @property
    def profit(self) -> float:
        return self.payout - self.amount


# =====================================================
# FRAGMENTOS GRAPHQL (basados en StakePy)
# =====================================================

# Fragmento CasinoBet - campos comunes para apuestas de casino
CASINO_BET_FRAGMENT = """
fragment CasinoBet on CasinoBet {
    id
    active
    payoutMultiplier
    amountMultiplier
    amount
    payout
    updatedAt
    currency
    game
    user {
        id
        name
    }
}
"""

# Fragmento para estado de juego Dice
CASINO_GAME_DICE_FRAGMENT = """
fragment CasinoGameDice on CasinoGameDice {
    result
    target
    condition
}
"""

# Fragmento para estado de juego Limbo
CASINO_GAME_LIMBO_FRAGMENT = """
fragment CasinoGameLimbo on CasinoGameLimbo {
    result
    multiplierTarget
}
"""

# Fragmento para apuestas deportivas (HighRollers)
# NOTA: El campo "odds" no existe en SportBet, usar payoutMultiplier
REALTIME_SPORT_BET_FRAGMENT = """
fragment RealtimeSportBet on Bet {
    id
    iid
    bet {
        ... on SportBet {
            id
            amount
            currency
            payout
            payoutMultiplier
            status
            user {
                id
                name
            }
        }
    }
}
"""


class StakeMonitor(BaseMonitor):
    """
    Monitor para Stake.com.

    NOTA: Stake tiene protecciones anti-bot significativas.
    - La API GraphQL requiere queries exactos (whitelisted)
    - Se necesita cf_clearance cookie para bypass de Cloudflare
    - Para casino, es mejor monitorear streams de Kick

    Este monitor proporciona:
    1. Query de UserBalances (requiere autenticación)
    2. HighRollers de deportes (público)
    3. Integración con KickMonitor para detectar Big Wins en streams
    """

    # GraphQL endpoints (stake.krd es mirror, stake.com es principal)
    GRAPHQL_URLS = [
        "https://stake.com/_api/graphql",
        "https://stake.krd/_api/graphql",
    ]

    # Query para obtener high rollers de deportes (funciona sin auth)
    # NOTA: Esta query está whitelisted en Stake y funciona públicamente
    HIGH_ROLLERS_SPORT_QUERY = """
    query highrollerSportBets($limit: Int!) {
        highrollerSportBets(limit: $limit) {
            ...RealtimeSportBet
            __typename
        }
    }
    """ + REALTIME_SPORT_BET_FRAGMENT

    # Query para balances de usuario (requiere x-access-token)
    USER_BALANCES_QUERY = """
    query UserBalances {
        user {
            id
            balances {
                available {
                    amount
                    currency
                    __typename
                }
                vault {
                    amount
                    currency
                    __typename
                }
                __typename
            }
            __typename
        }
    }
    """

    def __init__(
        self,
        big_win_threshold_usd: float = 10000.0,
        poll_interval_seconds: float = 5.0,
        use_cloudflare_bypass: bool = True,
        api_token: Optional[str] = None,
        cf_clearance: Optional[str] = None,
    ):
        """
        Args:
            big_win_threshold_usd: Umbral en USD para considerar Big Win
            poll_interval_seconds: Intervalo de polling
            use_cloudflare_bypass: Usar curl_cffi para bypass de Cloudflare
            api_token: Token de API de Stake (opcional, para queries autenticadas)
            cf_clearance: Cookie cf_clearance para bypass de Cloudflare
        """
        super().__init__(name="StakeMonitor")

        self.big_win_threshold = big_win_threshold_usd
        self.poll_interval = poll_interval_seconds
        self.use_bypass = use_cloudflare_bypass and HAS_CURL_CFFI
        self.api_token = api_token
        self.cf_clearance = cf_clearance

        self._http_client: Optional[httpx.AsyncClient] = None
        self._curl_session: Optional[Any] = None
        self._seen_bets: set = set()  # Para evitar duplicados
        self._last_poll_time: Optional[datetime] = None
        self._active_endpoint: str = self.GRAPHQL_URLS[0]

        # Headers para simular navegador
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/json",
            "Origin": "https://stake.com",
            "Referer": "https://stake.com/",
        }

        # Agregar token de API si está disponible
        if self.api_token:
            self._headers["x-access-token"] = self.api_token

    async def connect(self) -> bool:
        """Establece conexión con Stake."""
        try:
            if self.use_bypass:
                # Usar curl_cffi para bypass de Cloudflare
                self._curl_session = AsyncSession(impersonate="chrome")
            else:
                # Usar httpx estándar (puede ser bloqueado)
                self._http_client = httpx.AsyncClient(
                    headers=self._headers,
                    timeout=30.0,
                    follow_redirects=True,
                )

            # Verificar conexión probando diferentes endpoints
            success = await self._test_connection()
            self.is_connected = success
            return success

        except Exception as e:
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Cierra las conexiones."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._curl_session:
            # curl_cffi no tiene close async directo
            self._curl_session = None

        self.is_connected = False

    async def _test_connection(self) -> bool:
        """Prueba la conexión con diferentes endpoints."""
        for endpoint in self.GRAPHQL_URLS:
            try:
                # Probar con query de high rollers deportivos
                result = await self._execute_graphql(
                    self.HIGH_ROLLERS_SPORT_QUERY,
                    {"limit": 1},
                    endpoint=endpoint,
                )
                if result and "data" in result:
                    self._active_endpoint = endpoint
                    return True
            except Exception:
                continue

        # Si ninguno funciona, intentar sin datos (solo conexión)
        return False

    async def _execute_graphql(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ejecuta una query GraphQL contra Stake.

        Args:
            query: Query GraphQL
            variables: Variables de la query
            endpoint: URL del endpoint (usa _active_endpoint si no se especifica)

        Returns:
            Respuesta JSON o None si hay error
        """
        url = endpoint or self._active_endpoint
        payload = {
            "query": query,
            "variables": variables or {},
        }

        try:
            if self.use_bypass and self._curl_session:
                # Agregar cookies si están disponibles
                cookies = {}
                if self.cf_clearance:
                    cookies["cf_clearance"] = self.cf_clearance

                response = await self._curl_session.post(
                    url,
                    json=payload,
                    headers=self._headers,
                    cookies=cookies if cookies else None,
                )
                if response.status_code == 200:
                    return response.json()

            elif self._http_client:
                response = await self._http_client.post(
                    url,
                    json=payload,
                )
                if response.status_code == 200:
                    return response.json()

            return None

        except Exception as e:
            return None

    async def _listen(self) -> None:
        """
        Loop principal de monitoreo.

        NOTA: Stake tiene API con queries whitelisted.
        Usamos polling de high rollers de deportes como base.
        Para casino, se recomienda monitorear streams de Kick.
        """
        while self.is_running:
            try:
                await self._poll_high_rollers_sport()
                self._last_poll_time = datetime.now()
                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                await asyncio.sleep(self.poll_interval * 2)

    async def _poll_high_rollers_sport(self) -> None:
        """Consulta los High Rollers de deportes."""
        result = await self._execute_graphql(
            self.HIGH_ROLLERS_SPORT_QUERY,
            {"limit": 50}
        )

        if not result or "data" not in result:
            return

        bets_data = result.get("data", {}).get("highrollerSportBets", [])

        for bet_entry in bets_data:
            # Estructura de SportBet: { id, iid, bet: { ... on SportBet { ... } } }
            bet_data = bet_entry.get("bet", {})
            if not bet_data:
                continue

            bet_id = bet_data.get("id", "") or bet_entry.get("id", "")

            # Evitar procesar duplicados
            if bet_id in self._seen_bets:
                continue

            self._seen_bets.add(bet_id)

            # Limitar tamaño del set de bets vistos
            if len(self._seen_bets) > 10000:
                # Limpiar los más antiguos (aproximado)
                self._seen_bets = set(list(self._seen_bets)[-5000:])

            # Parsear datos de la apuesta (estructura SportBet)
            payout = float(bet_data.get("payout", 0) or 0)
            amount = float(bet_data.get("amount", 0) or 0)
            multiplier = float(bet_data.get("payoutMultiplier", 0) or 0)
            currency = bet_data.get("currency", "usd") or "usd"

            # Convertir a USD (simplificado - en producción usar rates reales)
            payout_usd = self._estimate_usd_value(payout, currency)

            # Verificar si es Big Win
            is_big_win = payout_usd >= self.big_win_threshold

            # Obtener username de forma segura
            user_data = bet_data.get("user")
            username = user_data.get("name", "Anonymous") if user_data else "Anonymous"

            bet = StakeBet(
                bet_id=bet_id,
                game="Sports Bet",
                user=username,
                amount=amount,
                payout=payout,
                multiplier=multiplier,
                currency=currency,
                timestamp=datetime.now(),
                is_big_win=is_big_win,
            )

            if is_big_win:
                await self._handle_big_win(bet)

    async def _handle_big_win(self, bet: StakeBet) -> None:
        """Procesa un Big Win detectado."""
        event = MonitorEvent(
            event_type=EventType.BIG_WIN,
            source="stake",
            channel=bet.game,
            timestamp=bet.timestamp,
            data={
                "bet_id": bet.bet_id,
                "game": bet.game,
                "user": bet.user,
                "amount": bet.amount,
                "payout": bet.payout,
                "multiplier": bet.multiplier,
                "currency": bet.currency,
                "profit": bet.profit,
            },
            metadata={
                "virality_signal": True,
                "signal_weight": 0.25,
                "estimated_usd": self._estimate_usd_value(bet.payout, bet.currency),
            }
        )

        await self._emit_event(event)

    def _estimate_usd_value(self, amount: float, currency: str) -> float:
        """
        Estima el valor en USD de una cantidad.

        NOTA: En producción, usar API de rates en tiempo real.
        Estos son valores aproximados para demo.
        """
        # Rates aproximados (actualizar periódicamente)
        rates = {
            "usd": 1.0,
            "btc": 43000.0,
            "eth": 2200.0,
            "ltc": 70.0,
            "doge": 0.08,
            "bch": 230.0,
            "xrp": 0.50,
            "trx": 0.10,
            "eos": 0.70,
            "bnb": 310.0,
        }

        currency_lower = currency.lower()
        rate = rates.get(currency_lower, 1.0)

        return amount * rate

    async def get_recent_big_wins(
        self,
        limit: int = 10,
        min_payout_usd: float = 5000.0
    ) -> List[StakeBet]:
        """
        Obtiene los Big Wins recientes de apuestas deportivas.

        Args:
            limit: Número máximo de resultados
            min_payout_usd: Pago mínimo en USD

        Returns:
            Lista de StakeBet
        """
        result = await self._execute_graphql(
            self.HIGH_ROLLERS_SPORT_QUERY,
            {"limit": limit * 2}  # Pedir más para filtrar
        )

        if not result or "data" not in result:
            return []

        bets = []
        bets_data = result.get("data", {}).get("highrollerSportBets", [])

        for bet_entry in bets_data:
            bet_data = bet_entry.get("bet", {})
            if not bet_data:
                continue

            payout = float(bet_data.get("payout", 0) or 0)
            currency = bet_data.get("currency", "usd") or "usd"
            payout_usd = self._estimate_usd_value(payout, currency)

            if payout_usd >= min_payout_usd:
                # Obtener username de forma segura
                user_data = bet_data.get("user")
                username = user_data.get("name", "Anonymous") if user_data else "Anonymous"

                bet = StakeBet(
                    bet_id=bet_data.get("id", "") or bet_entry.get("id", ""),
                    game="Sports Bet",
                    user=username,
                    amount=float(bet_data.get("amount", 0) or 0),
                    payout=payout,
                    multiplier=float(bet_data.get("payoutMultiplier", 0) or 0),
                    currency=currency,
                    timestamp=datetime.now(),
                    is_big_win=True,
                )
                bets.append(bet)

                if len(bets) >= limit:
                    break

        return bets


# =====================================================
# EJEMPLO DE USO
# =====================================================

async def main():
    """Ejemplo de uso del StakeMonitor."""
    from core.logger import logger

    logger.info("=== StakeMonitor - High Rollers Deportivos ===")
    logger.info("NOTA: Para casino (Crash, Dice, etc), usa KickMonitor")

    monitor = StakeMonitor(
        big_win_threshold_usd=10000.0,
        poll_interval_seconds=10.0,
    )

    # Handler para Big Wins
    async def on_big_win(event: MonitorEvent):
        data = event.data
        logger.info(
            f"[BIG WIN] {data['game']}\n"
            f"   User: {data['user']}\n"
            f"   Payout: {data['payout']:.2f} {data['currency'].upper()}\n"
            f"   Multiplier: {data['multiplier']:.2f}x"
        )

    monitor.on_event(EventType.BIG_WIN, on_big_win)

    try:
        logger.info("Conectando a Stake...")
        success = await monitor.connect()
        if not success:
            logger.warning("No se pudo conectar a la API de Stake.")
            logger.info("Esto puede deberse a:")
            logger.info("  - Bloqueo de Cloudflare (necesitas cf_clearance cookie)")
            logger.info("  - La query no esta whitelisted")
            logger.info("  - Necesitas VPN/proxy")
            logger.info("")
            logger.info("Alternativa: Usa KickMonitor para monitorear streams de Stake")
            return

        logger.info(f"Conectado a: {monitor._active_endpoint}")
        await monitor.start()

        # Obtener Big Wins recientes
        recent_wins = await monitor.get_recent_big_wins(limit=5)
        if recent_wins:
            logger.info(f"Big Wins recientes: {len(recent_wins)}")
            for win in recent_wins:
                logger.info(f"  - {win.game}: {win.payout:.2f} {win.currency} ({win.multiplier:.2f}x)")
        else:
            logger.info("No se encontraron Big Wins recientes (o API no respondio)")

        # Mantener corriendo
        logger.info("Monitoreando high rollers... (Ctrl+C para detener)")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Deteniendo monitor...")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
