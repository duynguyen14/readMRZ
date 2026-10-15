/*
    Async passport batch processing tables for SQL Server.

    Files are stored on disk. RelativeImagePath is relative to the configured
    passport upload root; image bytes are intentionally not stored in SQL Server.
*/

USE [HOCHIEU];
GO

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.readmrz_passport_batches', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_passport_batches (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_passport_batches PRIMARY KEY,

        BatchKey UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_BatchKey DEFAULT NEWSEQUENTIALID(),
        ClientRequestId NVARCHAR(128) NULL,
        SourceSystem NVARCHAR(128) NULL,
        RequestedBy NVARCHAR(128) NULL,

        Status NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_Status DEFAULT (N'receiving'),
        UploadCompleted BIT NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_UploadCompleted DEFAULT (0),

        TotalItems INT NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_TotalItems DEFAULT (0),
        QueuedItems INT NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_QueuedItems DEFAULT (0),
        ProcessingItems INT NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_ProcessingItems DEFAULT (0),
        SucceededItems INT NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_SucceededItems DEFAULT (0),
        FailedItems INT NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_FailedItems DEFAULT (0),
        CancelledItems INT NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_CancelledItems DEFAULT (0),

        StorageDirectory NVARCHAR(1000) NOT NULL,
        MetadataJson NVARCHAR(MAX) NULL,
        ErrorMessage NVARCHAR(2000) NULL,

        UploadCompletedAt DATETIME2(3) NULL,
        ProcessStartedAt DATETIME2(3) NULL,
        CompletedAt DATETIME2(3) NULL,
        CreatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_CreatedDate DEFAULT SYSUTCDATETIME(),
        UpdatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_passport_batches_UpdatedDate DEFAULT SYSUTCDATETIME(),

        RowVersion ROWVERSION NOT NULL,

        CONSTRAINT UQ_readmrz_passport_batches_BatchKey UNIQUE (BatchKey),

        CONSTRAINT CK_readmrz_passport_batches_Status
            CHECK (Status IN (
                N'receiving',
                N'queued',
                N'processing',
                N'completed',
                N'completed_with_errors',
                N'failed',
                N'cancelled'
            )),

        CONSTRAINT CK_readmrz_passport_batches_Counts
            CHECK (
                TotalItems >= 0
                AND QueuedItems >= 0
                AND ProcessingItems >= 0
                AND SucceededItems >= 0
                AND FailedItems >= 0
                AND CancelledItems >= 0
            ),

        CONSTRAINT CK_readmrz_passport_batches_MetadataJson
            CHECK (MetadataJson IS NULL OR ISJSON(MetadataJson) = 1)
    );
END;
GO

IF OBJECT_ID(N'dbo.readmrz_passport_items', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_passport_items (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_passport_items PRIMARY KEY,

        ItemKey UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT DF_readmrz_passport_items_ItemKey DEFAULT NEWSEQUENTIALID(),
        BatchId BIGINT NOT NULL,
        SequenceNo INT NOT NULL,

        OriginalFileName NVARCHAR(512) NOT NULL,
        RelativeImagePath NVARCHAR(1000) NOT NULL,
        ContentType NVARCHAR(128) NULL,
        FileSizeBytes BIGINT NULL,
        Sha256 NVARCHAR(64) NULL,
        ImageWidth INT NULL,
        ImageHeight INT NULL,

        Status NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_passport_items_Status DEFAULT (N'queued'),
        AttemptCount INT NOT NULL
            CONSTRAINT DF_readmrz_passport_items_AttemptCount DEFAULT (0),
        MaxAttempts INT NOT NULL
            CONSTRAINT DF_readmrz_passport_items_MaxAttempts DEFAULT (3),
        NextAttemptAt DATETIME2(3) NULL,
        LockedBy NVARCHAR(128) NULL,
        LockedAt DATETIME2(3) NULL,

        OrientationDegree INT NULL,
        YoloConfidence FLOAT NULL,
        OcrConfidence FLOAT NULL,
        UsedFallback BIT NOT NULL
            CONSTRAINT DF_readmrz_passport_items_UsedFallback DEFAULT (0),
        ProcessTimeMs INT NULL,

        MrzLineOne NVARCHAR(128) NULL,
        MrzLineTwo NVARCHAR(128) NULL,
        PassportType NVARCHAR(20) NULL,
        IssuingCountry NVARCHAR(20) NULL,
        Surname NVARCHAR(255) NULL,
        GivenNames NVARCHAR(255) NULL,
        FullName NVARCHAR(512) NULL,
        PassportNumber NVARCHAR(50) NULL,
        Nationality NVARCHAR(20) NULL,
        DateOfBirth DATE NULL,
        Sex NVARCHAR(20) NULL,
        DateOfExpiry DATE NULL,
        PersonalNumber NVARCHAR(100) NULL,
        ChecksumValid BIT NULL,

        ResultJson NVARCHAR(MAX) NULL,
        DebugJson NVARCHAR(MAX) NULL,
        ErrorCode NVARCHAR(100) NULL,
        ErrorMessage NVARCHAR(2000) NULL,

        ReviewStatus NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_passport_items_ReviewStatus DEFAULT (N'not_ready'),
        ReviewedResultJson NVARCHAR(MAX) NULL,
        ReviewNote NVARCHAR(2000) NULL,
        ReviewedBy NVARCHAR(128) NULL,
        ReviewedAt DATETIME2(3) NULL,
        ReviewUpdatedDate DATETIME2(3) NULL,

        ProcessStartedAt DATETIME2(3) NULL,
        ProcessedAt DATETIME2(3) NULL,
        CreatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_passport_items_CreatedDate DEFAULT SYSUTCDATETIME(),
        UpdatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_passport_items_UpdatedDate DEFAULT SYSUTCDATETIME(),

        RowVersion ROWVERSION NOT NULL,

        CONSTRAINT FK_readmrz_passport_items_Batch
            FOREIGN KEY (BatchId) REFERENCES dbo.readmrz_passport_batches(Id),

        CONSTRAINT UQ_readmrz_passport_items_ItemKey UNIQUE (ItemKey),
        CONSTRAINT UQ_readmrz_passport_items_BatchSequence UNIQUE (BatchId, SequenceNo),

        CONSTRAINT CK_readmrz_passport_items_Status
            CHECK (Status IN (
                N'queued',
                N'processing',
                N'retry_wait',
                N'succeeded',
                N'failed',
                N'cancelled'
            )),

        CONSTRAINT CK_readmrz_passport_items_Attempts
            CHECK (AttemptCount >= 0 AND MaxAttempts > 0 AND AttemptCount <= MaxAttempts),

        CONSTRAINT CK_readmrz_passport_items_FileSize
            CHECK (FileSizeBytes IS NULL OR FileSizeBytes >= 0),

        CONSTRAINT CK_readmrz_passport_items_ImageSize
            CHECK (
                (ImageWidth IS NULL OR ImageWidth > 0)
                AND (ImageHeight IS NULL OR ImageHeight > 0)
            ),

        CONSTRAINT CK_readmrz_passport_items_OrientationDegree
            CHECK (OrientationDegree IS NULL OR OrientationDegree IN (0, 90, 180, 270)),

        CONSTRAINT CK_readmrz_passport_items_RelativeImagePath
            CHECK (
                RelativeImagePath NOT LIKE N'[A-Za-z]:%'
                AND LEFT(RelativeImagePath, 1) NOT IN (N'/', N'\')
                AND RelativeImagePath NOT LIKE N'\\%'
            ),

        CONSTRAINT CK_readmrz_passport_items_Sha256
            CHECK (
                Sha256 IS NULL
                OR (LEN(Sha256) = 64 AND Sha256 NOT LIKE N'%[^0-9A-Fa-f]%')
            ),

        CONSTRAINT CK_readmrz_passport_items_ResultJson
            CHECK (ResultJson IS NULL OR ISJSON(ResultJson) = 1),

        CONSTRAINT CK_readmrz_passport_items_DebugJson
            CHECK (DebugJson IS NULL OR ISJSON(DebugJson) = 1),

        CONSTRAINT CK_readmrz_passport_items_ReviewStatus
            CHECK (ReviewStatus IN (
                N'not_ready',
                N'pending',
                N'approved',
                N'corrected',
                N'rejected'
            )),

        CONSTRAINT CK_readmrz_passport_items_ReviewedResultJson
            CHECK (ReviewedResultJson IS NULL OR ISJSON(ReviewedResultJson) = 1)
    );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_items_BatchReviewStatus'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
)
AND COL_LENGTH(N'dbo.readmrz_passport_items', N'ReviewStatus') IS NOT NULL
BEGIN
    CREATE INDEX IX_readmrz_passport_items_BatchReviewStatus
    ON dbo.readmrz_passport_items (BatchId, ReviewStatus, SequenceNo)
    INCLUDE (ItemKey, OriginalFileName, Status, ReviewedBy, ReviewedAt, ReviewUpdatedDate);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'UX_readmrz_passport_batches_ClientRequestId'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_batches')
)
BEGIN
    CREATE UNIQUE INDEX UX_readmrz_passport_batches_ClientRequestId
    ON dbo.readmrz_passport_batches (ClientRequestId)
    WHERE ClientRequestId IS NOT NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_batches_Status'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_batches')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_batches_Status
    ON dbo.readmrz_passport_batches (Status, CreatedDate, Id)
    INCLUDE (BatchKey, UploadCompleted, TotalItems, SucceededItems, FailedItems);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_items_WorkQueue'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_items_WorkQueue
    ON dbo.readmrz_passport_items (Status, NextAttemptAt, Id)
    INCLUDE (BatchId, RelativeImagePath, AttemptCount, MaxAttempts);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_items_BatchStatus'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_items_BatchStatus
    ON dbo.readmrz_passport_items (BatchId, Status, SequenceNo)
    INCLUDE (ItemKey, OriginalFileName, ProcessedAt, ErrorCode);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_items_Sha256'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_items_Sha256
    ON dbo.readmrz_passport_items (Sha256)
    WHERE Sha256 IS NOT NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_items_PassportNumber'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_items_PassportNumber
    ON dbo.readmrz_passport_items (PassportNumber, ProcessedAt DESC)
    INCLUDE (BatchId, ItemKey, FullName, Nationality, DateOfBirth, DateOfExpiry)
    WHERE PassportNumber IS NOT NULL;
END;
GO

IF OBJECT_ID(N'dbo.vw_readmrz_passport_batch_status', N'V') IS NULL
BEGIN
    EXEC(N'CREATE VIEW dbo.vw_readmrz_passport_batch_status AS SELECT 1 AS Placeholder;');
END;
GO

ALTER VIEW dbo.vw_readmrz_passport_batch_status
AS
SELECT
    batch.Id,
    batch.BatchKey,
    batch.ClientRequestId,
    batch.SourceSystem,
    batch.RequestedBy,
    batch.Status,
    batch.UploadCompleted,
    batch.StorageDirectory,
    COUNT(item.Id) AS ActualTotalItems,
    SUM(CASE WHEN item.Status = N'queued' THEN 1 ELSE 0 END) AS ActualQueuedItems,
    SUM(CASE WHEN item.Status = N'processing' THEN 1 ELSE 0 END) AS ActualProcessingItems,
    SUM(CASE WHEN item.Status = N'retry_wait' THEN 1 ELSE 0 END) AS ActualRetryWaitItems,
    SUM(CASE WHEN item.Status = N'succeeded' THEN 1 ELSE 0 END) AS ActualSucceededItems,
    SUM(CASE WHEN item.Status = N'failed' THEN 1 ELSE 0 END) AS ActualFailedItems,
    SUM(CASE WHEN item.Status = N'cancelled' THEN 1 ELSE 0 END) AS ActualCancelledItems,
    CAST(
        CASE
            WHEN COUNT(item.Id) = 0 THEN 0
            ELSE 100.0 * SUM(CASE WHEN item.Status IN (N'succeeded', N'failed', N'cancelled') THEN 1 ELSE 0 END)
                / COUNT(item.Id)
        END
        AS DECIMAL(5,2)
    ) AS ProgressPercent,
    batch.UploadCompletedAt,
    batch.ProcessStartedAt,
    batch.CompletedAt,
    batch.ErrorMessage,
    batch.CreatedDate,
    batch.UpdatedDate
FROM dbo.readmrz_passport_batches batch
LEFT JOIN dbo.readmrz_passport_items item
    ON item.BatchId = batch.Id
GROUP BY
    batch.Id,
    batch.BatchKey,
    batch.ClientRequestId,
    batch.SourceSystem,
    batch.RequestedBy,
    batch.Status,
    batch.UploadCompleted,
    batch.StorageDirectory,
    batch.UploadCompletedAt,
    batch.ProcessStartedAt,
    batch.CompletedAt,
    batch.ErrorMessage,
    batch.CreatedDate,
    batch.UpdatedDate;
GO

/*
    Useful reads after the API/worker starts writing data:

    SELECT *
    FROM dbo.vw_readmrz_passport_batch_status
    WHERE BatchKey = @BatchKey;

    SELECT *
    FROM dbo.readmrz_passport_items
    WHERE BatchId = @BatchId
    ORDER BY SequenceNo;
*/
