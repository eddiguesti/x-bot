"""Mock data generator for testing and development."""

import random
from datetime import datetime, timedelta
from typing import Optional

from .x_client import XPost


# Sample tweets from crypto traders (realistic examples)
SAMPLE_TWEETS = [
    # Bullish BTC
    ("trader1", "BTC looking extremely bullish here. Accumulating more. 🚀"),
    ("trader2", "Going long BTC at these levels. Support holding strong."),
    ("trader3", "Bitcoin breakout incoming. I'm positioned long."),
    ("trader4", "Buying more BTC here. This dip is a gift."),
    ("trader5", "I am long BTC from 85k. Target 100k."),

    # Bearish BTC
    ("trader6", "BTC looks weak. Shorting this rally."),
    ("trader7", "Bitcoin rejection at resistance. Short entered."),
    ("trader8", "Bearish on BTC short term. Expecting a dump."),

    # Bullish ETH
    ("trader1", "ETH/BTC looking ready to pump. Long ethereum."),
    ("trader2", "Accumulating ETH here. Bullish setup."),
    ("trader9", "Ethereum breakout soon. Going long."),

    # Bearish ETH
    ("trader3", "ETH looking bearish. Selling my position."),
    ("trader10", "Short ETH here. Expecting rejection."),

    # Neutral/No signal
    ("trader4", "Interesting day in crypto markets."),
    ("trader5", "Watching BTC closely here."),
    ("trader6", "Markets are wild today."),

    # Mixed signals
    ("trader7", "BTC could go either way here. Waiting for confirmation."),
    ("trader8", "If support holds, I'll go long BTC. Otherwise short."),
]


def generate_mock_posts(
    num_posts: int = 50,
    days_back: int = 7,
    usernames: Optional[list[str]] = None,
) -> list[XPost]:
    """
    Generate mock posts for testing.

    Args:
        num_posts: Number of posts to generate
        days_back: How many days back to distribute posts
        usernames: Optional list of usernames to use

    Returns:
        List of mock XPost objects
    """
    if usernames is None:
        usernames = [f"trader{i}" for i in range(1, 11)]

    posts = []
    now = datetime.utcnow()

    for i in range(num_posts):
        # Random time in the past
        hours_ago = random.uniform(0, days_back * 24)
        posted_at = now - timedelta(hours=hours_ago)

        # Pick a random tweet template
        username, text = random.choice(SAMPLE_TWEETS)

        # Use provided username if available
        if usernames:
            username = random.choice(usernames)

        # Add some variation to the text
        variations = [
            "",
            " 📈",
            " 📉",
            " Let's see how this plays out.",
            " NFA.",
            " DYOR.",
        ]
        text = text + random.choice(variations)

        post = XPost(
            post_id=f"mock_{i}_{int(posted_at.timestamp())}",
            username=username,
            text=text,
            posted_at=posted_at,
            url=f"https://x.com/{username}/status/{i}",
            raw_data={"mock": True, "index": i},
        )
        posts.append(post)

    # Sort by time (newest first)
    posts.sort(key=lambda p: p.posted_at, reverse=True)

    return posts


def generate_historical_signals(
    session,
    num_signals: int = 100,
    days_back: int = 30,
):
    """
    Generate historical mock signals with outcomes for backtesting.

    This creates signals that have already been evaluated, useful for
    testing the ranking and consensus systems.
    """
    from ..models import Creator, Signal, Asset, Direction, SignalOutcome
    import json

    now = datetime.utcnow()

    # Ensure we have creators
    creators = session.query(Creator).filter(Creator.is_active == True).all()
    if not creators:
        # Create some mock creators
        for i in range(10):
            creator = Creator(
                username=f"trader{i+1}",
                display_name=f"Trader {i+1}",
            )
            session.add(creator)
        session.commit()
        creators = session.query(Creator).all()

    # Assign different skill levels to creators
    # Some are good (70% accuracy), some are bad (30% accuracy)
    creator_skills = {}
    for i, creator in enumerate(creators):
        if i < 3:  # Top 3 are skilled
            creator_skills[creator.id] = 0.70
        elif i < 6:  # Middle are average
            creator_skills[creator.id] = 0.50
        else:  # Bottom are poor
            creator_skills[creator.id] = 0.35

    signals_created = 0

    for i in range(num_signals):
        # Random time
        hours_ago = random.uniform(24, days_back * 24)  # At least 24h ago
        posted_at = now - timedelta(hours=hours_ago)
        eval_time = posted_at + timedelta(hours=24)

        # Pick random creator and asset
        creator = random.choice(creators)
        asset = random.choice([Asset.BTC, Asset.ETH])
        direction = random.choice([Direction.LONG, Direction.SHORT])

        # Determine outcome based on creator skill
        skill = creator_skills.get(creator.id, 0.5)
        is_correct = random.random() < skill

        # Generate realistic price data
        if asset == Asset.BTC:
            base_price = random.uniform(80000, 90000)
        else:
            base_price = random.uniform(2500, 3500)

        # Price change based on outcome
        if is_correct:
            if direction == Direction.LONG:
                change = random.uniform(0.5, 5.0)
            else:
                change = random.uniform(-5.0, -0.5)
        else:
            if direction == Direction.LONG:
                change = random.uniform(-5.0, -0.5)
            else:
                change = random.uniform(0.5, 5.0)

        price_at_eval = base_price * (1 + change / 100)

        # Create signal
        signal = Signal(
            creator_id=creator.id,
            post_id=f"hist_{i}_{int(posted_at.timestamp())}",
            post_text=f"Mock historical signal: {direction.value} {asset.value}",
            posted_at=posted_at,
            asset=asset,
            direction=direction,
            confidence=random.uniform(0.7, 1.0),
            outcome=SignalOutcome.CORRECT if is_correct else SignalOutcome.INCORRECT,
            price_at_signal=base_price,
            price_at_evaluation=price_at_eval,
            price_change_percent=change,
            evaluated_at=eval_time,
            raw_data=json.dumps({"mock": True}),
        )
        session.add(signal)

        # Update creator stats
        creator.total_predictions += 1
        if is_correct:
            creator.correct_predictions += 1
        creator.last_prediction_at = posted_at

        signals_created += 1

    session.commit()

    # Update creator ratings based on their performance
    from ..scoring.glicko2 import Glicko2Calculator, Glicko2Rating

    calc = Glicko2Calculator()

    for creator in creators:
        # Get creator's signals
        creator_signals = session.query(Signal).filter(
            Signal.creator_id == creator.id
        ).all()

        # Build outcomes list
        outcomes = [
            (s.outcome == SignalOutcome.CORRECT, None)
            for s in creator_signals
        ]

        if outcomes:
            # Update rating
            rating = Glicko2Rating(
                rating=creator.rating,
                rd=creator.rating_deviation,
                volatility=creator.volatility,
            )
            new_rating = calc.update_rating(rating, outcomes)

            creator.rating = new_rating.rating
            creator.rating_deviation = new_rating.rd
            creator.volatility = new_rating.volatility

    session.commit()

    return signals_created
