/*
    Add human-review state to asynchronous passport batch items.

    This migration is idempotent and can be run more than once.
    AI output remains in ResultJson. Human-reviewed output is stored separately
    in ReviewedResultJson so the original model result remains auditable.
*/

USE [HOCHIEU];
GO

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.readmrz_passport_items', N'U') IS NULL
BEGIN
    THROW 50001, 'Table dbo.readmrz_passport_items does not exist. Run create_passport_batch_tables.sql first.', 1;
END;
GO

IF COL_LENGTH(N'dbo.readmrz_passport_items', N'ReviewStatus') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_passport_items
    ADD ReviewStatus NVARCHAR(32) NOT NULL
        CONSTRAINT DF_readmrz_passport_items_ReviewStatus DEFAULT (N'not_ready');
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.default_constraints
    WHERE parent_object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
      AND parent_column_id = COLUMNPROPERTY(
          OBJECT_ID(N'dbo.readmrz_passport_items'),
          N'ReviewStatus',
          N'ColumnId'
      )
)
BEGIN
    ALTER TABLE dbo.readmrz_passport_items
    ADD CONSTRAINT DF_readmrz_passport_items_ReviewStatus
        DEFAULT (N'not_ready') FOR ReviewStatus;
END;

IF COL_LENGTH(N'dbo.readmrz_passport_items', N'ReviewedResultJson') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_passport_items
    ADD ReviewedResultJson NVARCHAR(MAX) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_passport_items', N'ReviewNote') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_passport_items
    ADD ReviewNote NVARCHAR(2000) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_passport_items', N'ReviewedBy') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_passport_items
    ADD ReviewedBy NVARCHAR(128) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_passport_items', N'ReviewedAt') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_passport_items
    ADD ReviewedAt DATETIME2(3) NULL;
END;

IF COL_LENGTH(N'dbo.readmrz_passport_items', N'ReviewUpdatedDate') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_passport_items
    ADD ReviewUpdatedDate DATETIME2(3) NULL;
END;
GO

/*
    ReviewStatus is referenced below. Keep this GO boundary so SQL Server
    recompiles the next batch after ALTER TABLE has added the column.
*/

BEGIN TRANSACTION;

/* Existing successful OCR items are ready for human review. */
UPDATE dbo.readmrz_passport_items
SET ReviewStatus = N'pending',
    ReviewUpdatedDate = COALESCE(ReviewUpdatedDate, SYSUTCDATETIME())
WHERE Status = N'succeeded'
  AND ReviewStatus = N'not_ready';

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_readmrz_passport_items_ReviewStatus'
      AND parent_object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
)
BEGIN
    ALTER TABLE dbo.readmrz_passport_items WITH CHECK
    ADD CONSTRAINT CK_readmrz_passport_items_ReviewStatus
        CHECK (ReviewStatus IN (
            N'not_ready',
            N'pending',
            N'approved',
            N'corrected',
            N'rejected'
        ));
END;

IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'CK_readmrz_passport_items_ReviewedResultJson'
      AND parent_object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
)
BEGIN
    ALTER TABLE dbo.readmrz_passport_items WITH CHECK
    ADD CONSTRAINT CK_readmrz_passport_items_ReviewedResultJson
        CHECK (ReviewedResultJson IS NULL OR ISJSON(ReviewedResultJson) = 1);
END;

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_items_BatchReviewStatus'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_items_BatchReviewStatus
    ON dbo.readmrz_passport_items (BatchId, ReviewStatus, SequenceNo)
    INCLUDE (
        ItemKey,
        OriginalFileName,
        Status,
        ReviewedBy,
        ReviewedAt,
        ReviewUpdatedDate
    );
END;

COMMIT TRANSACTION;
GO

SELECT
    ReviewStatus,
    COUNT(*) AS ItemCount
FROM dbo.readmrz_passport_items
GROUP BY ReviewStatus
ORDER BY ReviewStatus;
GO
