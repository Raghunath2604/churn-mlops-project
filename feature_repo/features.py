from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

customer = Entity(name="customer_id", description="A customer identifier")
churn_source = FileSource(path="data/churn_features.parquet", timestamp_field="event_timestamp", created_timestamp_column="created")
churn_features_view = FeatureView(
    name="churn_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="tenure_months", dtype=Float32),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="total_charges", dtype=Float32),
        Field(name="num_support_tickets", dtype=Int64),
    ],
    source=churn_source,
)
