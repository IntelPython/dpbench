# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Init

Revision ID: 6477ee64f532
Revises:
Create Date: 2023-04-06 16:59:44.465528

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "6477ee64f532"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.Integer(),
            server_default=sa.text("(strftime('%s','now'))"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "results",
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("benchmark", sa.String(), nullable=False),
        sa.Column("implementation", sa.String(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("framework_version", sa.String(), nullable=False),
        sa.Column("error_state", sa.String(), nullable=False),
        sa.Column("problem_preset", sa.String(), nullable=False),
        sa.Column("input_size", sa.Integer(), nullable=False),
        sa.Column(
            "input_size_human",
            sa.Integer(),
            sa.Computed(
                "CASE WHEN (input_size >= 1024 AND input_size < 1048576) THEN (input_size / 1024) || 'KB' WHEN (input_size >= 1048576 AND input_size < 1073741824) THEN (input_size / 1048576) || 'MB' WHEN (input_size >= 1073741824 AND input_size < 1099511627776) THEN (input_size / 1073741824) || 'GB' WHEN (input_size >= 1099511627776 AND input_size < 1125899906842624) THEN (input_size / 1099511627776) || 'TB' ELSE input_size || 'B' END",
                persisted=False,
            ),
            nullable=False,
        ),
        sa.Column("setup_time", sa.Float(), nullable=False),
        sa.Column("warmup_time", sa.Float(), nullable=False),
        sa.Column("repeats", sa.String(), nullable=False),
        sa.Column("min_exec_time", sa.Float(), nullable=False),
        sa.Column("max_exec_time", sa.Float(), nullable=False),
        sa.Column("median_exec_time", sa.Float(), nullable=False),
        sa.Column("quartile25_exec_time", sa.Float(), nullable=False),
        sa.Column("quartile75_exec_time", sa.Float(), nullable=False),
        sa.Column("teardown_time", sa.Float(), nullable=False),
        sa.Column("validated", sa.String(), nullable=False),
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
        sa.UniqueConstraint("run_id", "benchmark", "implementation"),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("results")
    op.drop_table("runs")
    # ### end Alembic commands ###
