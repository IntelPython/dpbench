# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Add postfixes

Revision ID: c1afe59771e9
Revises: 6477ee64f532
Create Date: 2023-06-20 19:55:37.386101

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c1afe59771e9"
down_revision = "6477ee64f532"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "postfixes",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("postfix", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("device", sa.String(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.Integer(),
            server_default=sa.text("(strftime('%s','now'))"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "postfix"),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("postfixes")
    # ### end Alembic commands ###
