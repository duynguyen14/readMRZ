SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.readmrz_passport_evaluation_items', N'U') IS NULL
    THROW 50001, 'Missing dbo.readmrz_passport_evaluation_items.', 1;
GO

IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ReviewStatus') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD ReviewStatus NVARCHAR(32) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ReviewBasis') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD ReviewBasis NVARCHAR(64) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ReviewNote') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD ReviewNote NVARCHAR(2000) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ReviewedBy') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD ReviewedBy NVARCHAR(255) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ReviewedAt') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD ReviewedAt DATETIME2(3) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'ReviewUpdatedAt') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items ADD ReviewUpdatedAt DATETIME2(3) NULL;
IF COL_LENGTH(N'dbo.readmrz_passport_evaluation_items', N'IsPoorQualityImage') IS NULL
    ALTER TABLE dbo.readmrz_passport_evaluation_items
        ADD IsPoorQualityImage BIT NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_items_IsPoorQualityImage DEFAULT (0) WITH VALUES;
GO

/*
Strict automatic approval requires all three independent conditions:
  1. the newly-read MRZ passes ICAO checksum;
  2. every comparable field matches the source record;
  3. the source MRZ itself also passes ICAO checksum.
Everything else that has been source-compared is queued for human review.
Existing human decisions are deliberately preserved.
*/
UPDATE item
SET ReviewStatus = CASE
        WHEN item.ChecksumValid = 1
         AND item.AllComparedFieldsMatch = 1
         AND JSON_VALUE(item.SourceDataJson, '$.normalized_passport_fields.source_mrz_checksum_valid') = N'true'
            THEN N'auto_approved'
        ELSE N'pending'
    END,
    ReviewBasis = CASE
        WHEN item.ChecksumValid = 1
         AND item.AllComparedFieldsMatch = 1
         AND JSON_VALUE(item.SourceDataJson, '$.normalized_passport_fields.source_mrz_checksum_valid') = N'true'
            THEN N'strict_checksum_and_source_match'
        ELSE N'requires_human_review'
    END,
    ReviewUpdatedAt = COALESCE(item.ReviewUpdatedAt, SYSUTCDATETIME())
FROM dbo.readmrz_passport_evaluation_items item
WHERE item.SourceComparedAt IS NOT NULL
  AND (item.ReviewStatus IS NULL OR item.ReviewStatus IN (N'auto_approved', N'pending'));
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.check_constraints
    WHERE name = N'CK_readmrz_passport_evaluation_items_ReviewStatus'
)
BEGIN
    ALTER TABLE dbo.readmrz_passport_evaluation_items WITH CHECK
    ADD CONSTRAINT CK_readmrz_passport_evaluation_items_ReviewStatus CHECK (
        ReviewStatus IS NULL OR ReviewStatus IN (
            N'auto_approved', N'pending', N'approved', N'rejected'
        )
    );
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_evaluation_items_Review'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_evaluation_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_evaluation_items_Review
        ON dbo.readmrz_passport_evaluation_items (RunId, ReviewStatus, CountryCode, SequenceNo)
        INCLUDE (ItemKey, ChecksumValid, AllComparedFieldsMatch, SourceComparedAt);
END;
GO

SELECT
    RunId,
    COUNT_BIG(*) AS TotalCompared,
    SUM(CASE WHEN ReviewStatus = N'auto_approved' THEN 1 ELSE 0 END) AS AutoApproved,
    SUM(CASE WHEN ReviewStatus = N'pending' THEN 1 ELSE 0 END) AS PendingReview,
    SUM(CASE WHEN ReviewStatus = N'approved' THEN 1 ELSE 0 END) AS HumanApproved,
    SUM(CASE WHEN ReviewStatus = N'rejected' THEN 1 ELSE 0 END) AS Rejected
FROM dbo.readmrz_passport_evaluation_items
WHERE SourceComparedAt IS NOT NULL
GROUP BY RunId
ORDER BY RunId;
GO
