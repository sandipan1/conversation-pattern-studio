"""Create a reproducible synthetic dataset and an illustrative dashboard bundle."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = [
    (
        "Account access and team management",
        [
            ("Reset a password", "reset my account password"),
            ("Find a missing reset email", "find a password reset email that never arrived"),
            ("Sign in with SSO", "sign in with my company's SSO"),
            ("Set up two-factor authentication", "set up two-factor authentication"),
            ("Unlock an account", "unlock my account after too many sign-in attempts"),
            ("Invite a teammate", "invite a new teammate"),
            ("Change a member role", "change a teammate's role"),
            ("Remove a teammate", "remove a former teammate from our workspace"),
        ],
    ),
    (
        "Billing and subscriptions",
        [
            ("Download an invoice", "download an invoice"),
            ("Update a payment card", "update the payment card on our account"),
            ("Fix a declined payment", "fix a declined subscription payment"),
            ("Change a plan", "switch to a different subscription plan"),
            ("Adjust paid seats", "change our number of paid seats"),
            ("Understand usage charges", "understand our usage charges"),
            ("Add a tax ID", "add our tax ID to future invoices"),
            ("Cancel a subscription", "cancel our subscription"),
        ],
    ),
    (
        "Data and integrations",
        [
            ("Export workspace data", "export all our workspace data"),
            ("Import a CSV file", "import records from a CSV file"),
            ("Create an API key", "create an API key for an integration"),
            ("Handle API rate limits", "handle API rate limits in our app"),
            ("Configure webhooks", "configure webhook delivery"),
            ("Connect a CRM", "connect our CRM integration"),
            ("Search older conversations", "search older conversations"),
            ("Set data retention", "set our data retention period"),
        ],
    ),
]
QUESTION_TEMPLATES = [
    "I need to {action}. What should I do?",
    "Can you walk me through how to {action}?",
    "We're trying to {action} for our workspace.",
    "What is the quickest way to {action}?",
    "I'm stuck while trying to {action}. Can you help?",
]
ANCHORS = [(-9, -3), (9, -3), (0, 10)]


def build() -> tuple[list[dict], dict]:
    conversations: list[dict] = []
    summaries: list[dict] = []
    clusters: list[dict] = []
    parents: list[dict] = []
    for topic_index, (topic, needs) in enumerate(TOPICS):
        topic_chat_ids: list[str] = []
        for need_index, (name, action) in enumerate(needs):
            chat_ids: list[str] = []
            for template in QUESTION_TEMPLATES:
                chat_id = f"demo-{len(conversations) + 1:03d}"
                question = template.format(action=action)
                conversations.append({
                    "chat_id": chat_id,
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": f"I can help you {action}. What have you tried so far?"},
                    ],
                    "metadata": {"channel": "support" if len(conversations) % 2 == 0 else "web"},
                })
                summaries.append({"chat_id": chat_id, "summary": question, "request": action})
                chat_ids.append(chat_id)
                topic_chat_ids.append(chat_id)
            angle = 2 * math.pi * need_index / len(needs)
            anchor_x, anchor_y = ANCHORS[topic_index]
            clusters.append({
                "id": f"demo-pattern-{topic_index + 1}-{need_index + 1}",
                "name": name,
                "description": f"Conversations about how to {action}.",
                "slug": name.lower().replace(" ", "_"),
                "chat_ids": chat_ids,
                "parent_id": f"demo-theme-{topic_index + 1}",
                "level": 0,
                "x_coord": anchor_x + 3 * math.cos(angle),
                "y_coord": anchor_y + 3 * math.sin(angle),
            })
        parents.append({
            "id": f"demo-theme-{topic_index + 1}",
            "name": topic,
            "description": f"Related requests about {topic.lower()}.",
            "slug": topic.lower().replace(" ", "_"),
            "chat_ids": topic_chat_ids,
            "parent_id": "demo-theme-all",
            "level": 1,
            "x_coord": ANCHORS[topic_index][0],
            "y_coord": ANCHORS[topic_index][1],
        })
    root = {
        "id": "demo-theme-all",
        "name": "Workspace operations",
        "description": "Questions about access, billing, data, and integrations.",
        "slug": "workspace_operations",
        "chat_ids": [item["chat_id"] for item in conversations],
        "parent_id": None,
        "level": 2,
        "x_coord": 0,
        "y_coord": 0,
    }
    meta_clusters = [*parents, root]
    bundle = {
        "conversations": conversations,
        "summaries": summaries,
        "clusters": clusters,
        "meta_clusters": meta_clusters,
        "dimensionality": [*clusters, *meta_clusters],
    }
    return conversations, bundle


if __name__ == "__main__":
    conversations, bundle = build()
    dataset_path = ROOT / "examples" / "hierarchy-conversations.jsonl"
    bundle_path = ROOT / "pattern_studio" / "static" / "demo-analysis.json"
    dataset_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in conversations) + "\n",
        encoding="utf-8",
    )
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {len(conversations)} conversations and {len(bundle['dimensionality'])} demo groups")
