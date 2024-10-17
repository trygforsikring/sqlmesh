from __future__ import annotations

import logging
import re
import typing as t
from typing import List


from sqlglot import exp

from sqlmesh.core.engine_adapter.base_postgres import BasePostgresEngineAdapter
from sqlmesh.core.engine_adapter.mixins import (
    GetCurrentCatalogFromFunctionMixin,
    PandasNativeFetchDFSupportMixin,
)
from sqlmesh.core.engine_adapter.shared import set_catalog, CatalogSupport, CommentCreationView, CommentCreationTable
from sqlmesh.core.schema_diff import SchemaDiffer


if t.TYPE_CHECKING:
    from sqlmesh.core._typing import TableName
    from sqlmesh.core.engine_adapter._typing import DF

logger = logging.getLogger(__name__)

@set_catalog()
class RisingwaveEngineAdapter(
    BasePostgresEngineAdapter,
    PandasNativeFetchDFSupportMixin,
    GetCurrentCatalogFromFunctionMixin,
):
    DIALECT = "risingwave"
    SUPPORTS_INDEXES = True
    HAS_VIEW_BINDING = True
    CURRENT_CATALOG_EXPRESSION = exp.column("current_catalog")
    SUPPORTS_REPLACE_TABLE = False
    DEFAULT_BATCH_SIZE = 400
    CATALOG_SUPPORT = CatalogSupport.SINGLE_CATALOG_ONLY
    COMMENT_CREATION_TABLE = CommentCreationTable.COMMENT_COMMAND_ONLY
    COMMENT_CREATION_VIEW = CommentCreationView.COMMENT_COMMAND_ONLY
    SUPPORTS_MATERIALIZED_VIEWS = True

    SCHEMA_DIFFER = SchemaDiffer(
        parameterized_type_defaults={
            # DECIMAL without precision is "up to 131072 digits before the decimal point; up to 16383 digits after the decimal point"
            exp.DataType.build("DECIMAL", dialect=DIALECT).this: [(131072 + 16383, 16383), (0,)],
            exp.DataType.build("CHAR", dialect=DIALECT).this: [(1,)],
            exp.DataType.build("TIME", dialect=DIALECT).this: [(6,)],
            exp.DataType.build("TIMESTAMP", dialect=DIALECT).this: [(6,)],
        },
        types_with_unlimited_length={
            # all can ALTER to `TEXT`
            exp.DataType.build("TEXT", dialect=DIALECT).this: {
                exp.DataType.build("VARCHAR", dialect=DIALECT).this,
                exp.DataType.build("CHAR", dialect=DIALECT).this,
                exp.DataType.build("BPCHAR", dialect=DIALECT).this,
            },
            # all can ALTER to unparameterized `VARCHAR`
            exp.DataType.build("VARCHAR", dialect=DIALECT).this: {
                exp.DataType.build("VARCHAR", dialect=DIALECT).this,
                exp.DataType.build("CHAR", dialect=DIALECT).this,
                exp.DataType.build("BPCHAR", dialect=DIALECT).this,
                exp.DataType.build("TEXT", dialect=DIALECT).this,
            },
            # parameterized `BPCHAR(n)` can ALTER to unparameterized `BPCHAR`
            exp.DataType.build("BPCHAR", dialect=DIALECT).this: {
                exp.DataType.build("BPCHAR", dialect=DIALECT).this
            },
        },
    )

    def _fetch_native_df(
        self, query: t.Union[exp.Expression, str], quote_identifiers: bool = False
    ) -> DF:
        """
        `read_sql_query` when using psycopg will result on a hanging transaction that must be committed

        https://github.com/pandas-dev/pandas/pull/42277
        """
        df = super()._fetch_native_df(query, quote_identifiers)
        if not self._connection_pool.is_transaction_active:
            self._connection_pool.commit()
        return df


    def create_view(
        self,
        view_name: TableName,
        query_or_df: DF,
        columns_to_types: t.Optional[t.Dict[str, exp.DataType]] = None,
        replace: bool = True,
        materialized: bool = False,
        table_description: t.Optional[str] = None,
        column_descriptions: t.Optional[t.Dict[str, str]] = None,
        view_properties: t.Optional[t.Dict[str, exp.Expression]] = None,
        **create_kwargs: t.Any,
    ) -> None:
        """
        Postgres has very strict rules around view replacement. For example the new query must generate an identical setFormatter
        of columns, using the same column names and data types as the old one. We have to delete the old view instead of replacing it
        to work around these constraints.

        Reference: https://www.postgresql.org/docs/current/sql-createview.html
        """
        if materialized:
            replace = True

        with self.transaction():
            """check this why replace is always false incase of materialized is enabled to true"""
            if replace:
                self.drop_view(view_name, materialized=materialized)
            super().create_view(
                view_name,
                query_or_df,
                columns_to_types=columns_to_types,
                replace=replace,
                materialized=materialized,
                table_description=table_description,
                column_descriptions=column_descriptions,
                view_properties=view_properties,
                **create_kwargs,
            )


    def drop_view(
        self,
        view_name: TableName,
        ignore_if_not_exists: bool = True,
        materialized: bool = False,
        **kwargs: t.Any,
    ) -> None:
        kwargs["cascade"] = kwargs.get("cascade", True)

        return super().drop_view(
            view_name,
            ignore_if_not_exists=ignore_if_not_exists,
            materialized=self._is_materialized(view_name),
            **kwargs,
        )

    def _is_materialized(self, view_name: TableName) -> bool:
        _is_materialized = False
        """
           Builds a SQL query to check for a table in information_schema.tables
           based on dbname.schema_name.table_name.
           """

        # Build the base query with WHERE conditions
        query = (
            exp.select("table_type")
            .from_("information_schema.tables")
            .where(exp.column("table_schema").eq(view_name.db))
            .where(exp.column("table_name").eq(view_name.name))
        )

        # Fetch the result as a DataFrame
        df = self.fetchdf(query)

        if not df.empty:
            # Access the first row to get table_type
            first_row = df.iloc[0]
            table_type = first_row['table_type']

            # Check if table_type is MATERIALIZED VIEW
            if table_type == 'MATERIALIZED VIEW':
                _is_materialized = True
                logger.debug("The object is a MATERIALIZED VIEW.")
            else:
                logger.debug(f"The object is of type: {table_type}")
        else:
            logger.debug("The DataFrame is empty.")

        # Return whether the object is materialized
        return _is_materialized
