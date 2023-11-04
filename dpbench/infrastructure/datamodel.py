# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sqlite3

from alembic import command
from alembic.config import Config
from sqlalchemy import (
    Column,
    Computed,
    Engine,
    ForeignKey,
    UniqueConstraint,
    and_,
    case,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[int] = mapped_column(
        server_default=text("strftime('%s','now')")
    )
    pass


class Run(Base):
    __tablename__ = "runs"

    # results: Mapped[list["Result"]] = relationship(back_populates="run")


class Result(Base):
    __tablename__ = "results"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    # run: Mapped["Run"] = relationship()
    # run: Mapped["Run"] = relationship(back_populates="results")
    benchmark: Mapped[str]  # = mapped_column(primary_key=True)
    implementation: Mapped[str]  # = mapped_column(primary_key=True)
    platform: Mapped[str]
    framework_version: Mapped[str]
    error_state: Mapped[str]
    problem_preset: Mapped[str]
    input_size: Mapped[int]
    input_size_human: Mapped[int] = mapped_column(
        Computed(
            case(
                (
                    and_(
                        Column("input_size") >= 1024,
                        Column("input_size") < 1024 * 1024,
                    ),
                    Column("input_size").op("/")(1024).op("||")("KB"),
                ),
                (
                    and_(
                        Column("input_size") >= 1024 * 1024,
                        Column("input_size") < 1024 * 1024 * 1024,
                    ),
                    Column("input_size").op("/")(1024 * 1024).op("||")("MB"),
                ),
                (
                    and_(
                        Column("input_size") >= 1024 * 1024 * 1024,
                        Column("input_size") < 1024 * 1024 * 1024 * 1024,
                    ),
                    Column("input_size")
                    .op("/")(1024 * 1024 * 1024)
                    .op("||")("GB"),
                ),
                (
                    and_(
                        Column("input_size") >= 1024 * 1024 * 1024 * 1024,
                        Column("input_size") < 1024 * 1024 * 1024 * 1024 * 1024,
                    ),
                    Column("input_size")
                    .op("/")(1024 * 1024 * 1024 * 1024)
                    .op("||")("TB"),
                ),
                else_=Column("input_size").op("||")("B"),
            ),
            persisted=False,
        )
    )
    setup_time: Mapped[float]
    warmup_time: Mapped[float]
    repeats: Mapped[str]
    min_exec_time: Mapped[float]
    max_exec_time: Mapped[float]
    median_exec_time: Mapped[float]
    quartile25_exec_time: Mapped[float]
    quartile75_exec_time: Mapped[float]
    teardown_time: Mapped[float]
    validated: Mapped[str]

    __table_args__ = (
        UniqueConstraint("run_id", "benchmark", "implementation"),
    )


class Postfix(Base):
    __tablename__ = "postfixes"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    postfix: Mapped[str]
    description: Mapped[str]
    device: Mapped[str]

    __table_args__ = (UniqueConstraint("run_id", "postfix"),)


def create_connection(db_file) -> Engine:
    """create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        engine = create_engine(f"sqlite:///{db_file}")

        return engine
    except sqlite3.Error:
        logging.exception(
            "Failed to create a connection to database specified as " + db_file
        )

    return conn


def create_run(conn: Engine) -> int:
    """creates run record of the benchmarks and returns it's id
    :param conn: sqlalchemy engine
    :return: id of the run
    """
    run = Run()

    with Session(conn) as session:
        session.add(run)
        session.commit()

        return run.id


def create_results_table(db_file: str):
    """create sqlite database file and runs migrations to create all necessery tables.
    If file exists - it just updates it to the head version.
    :return:
    """

    absolute_path = os.path.dirname(__file__)
    relative_path = "../migrations/alembic.ini"
    full_path = os.path.join(absolute_path, relative_path)

    alembic_cfg = Config(full_path)
    alembic_cfg.set_main_option("sqlalchemy.url", "sqlite:///" + db_file)

    command.upgrade(alembic_cfg, "head")


def store_results(conn: Engine, result: Result):
    """creates result record in database.
    :param conn: sqlalchemy engine
    :param result: result record to be inserted into db
    :return:
    """
    with Session(conn) as session:
        session.add(result)
        session.commit()


def store_postfix(conn: Engine, postfix: Postfix):
    """creates postfix record in database.
    :param conn: sqlalchemy engine
    :param postfix: postfix record to be inserted into db
    :return:
    """
    with Session(conn) as session:
        existing_postfix = (
            session.query(Postfix)
            .filter_by(run_id=postfix.run_id, postfix=postfix.postfix)
            .first()
        )
        if existing_postfix is None:
            session.add(postfix)
            session.commit()
