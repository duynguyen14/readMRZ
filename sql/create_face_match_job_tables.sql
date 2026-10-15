/*
    Asynchronous passport face matching jobs for SQL Server.

    Original images and embedding files are stored on disk. SQL Server stores
    relative paths, processing state, compact extraction results and pairs.
*/

USE [HOCHIEU];
GO

SET NOCOUNT ON;
SET XACT_ABORT ON;
GO

IF OBJECT_ID(N'dbo.readmrz_face_match_jobs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_face_match_jobs (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_face_match_jobs PRIMARY KEY,
        JobKey UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT DF_readmrz_face_match_jobs_JobKey DEFAULT NEWSEQUENTIALID(),
        ClientRequestId NVARCHAR(128) NULL,
        SourceSystem NVARCHAR(128) NULL,
        RequestedBy NVARCHAR(128) NULL,
        Status NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_face_match_jobs_Status DEFAULT (N'receiving'),
        UploadCompleted BIT NOT NULL
            CONSTRAINT DF_readmrz_face_match_jobs_UploadCompleted DEFAULT (0),

        TotalItems INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Total DEFAULT (0),
        QueuedItems INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Queued DEFAULT (0),
        ProcessingItems INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Processing DEFAULT (0),
        SucceededItems INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Succeeded DEFAULT (0),
        FailedItems INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Failed DEFAULT (0),
        FaceItems INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Faces DEFAULT (0),
        PassportItems INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Passports DEFAULT (0),
        UnknownItems INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Unknown DEFAULT (0),
        MatchedPairs INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Matched DEFAULT (0),
        ReviewPairs INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_Review DEFAULT (0),
        UnmatchedFaces INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_UnmatchedFaces DEFAULT (0),
        UnmatchedPassports INT NOT NULL CONSTRAINT DF_readmrz_face_match_jobs_UnmatchedPassports DEFAULT (0),

        MatchThreshold FLOAT NOT NULL,
        ReviewThreshold FLOAT NOT NULL,
        StorageDirectory NVARCHAR(1000) NOT NULL,
        MetadataJson NVARCHAR(MAX) NULL,
        ErrorMessage NVARCHAR(2000) NULL,
        MatchLockedBy NVARCHAR(128) NULL,
        MatchLockedAt DATETIME2(3) NULL,

        UploadCompletedAt DATETIME2(3) NULL,
        ProcessStartedAt DATETIME2(3) NULL,
        MatchingStartedAt DATETIME2(3) NULL,
        CompletedAt DATETIME2(3) NULL,
        CreatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_face_match_jobs_Created DEFAULT SYSUTCDATETIME(),
        UpdatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_face_match_jobs_Updated DEFAULT SYSUTCDATETIME(),
        RowVersion ROWVERSION NOT NULL,

        CONSTRAINT UQ_readmrz_face_match_jobs_JobKey UNIQUE (JobKey),
        CONSTRAINT CK_readmrz_face_match_jobs_Status CHECK (Status IN (
            N'receiving', N'queued', N'extracting', N'matching',
            N'completed', N'completed_with_errors', N'failed', N'cancelled'
        )),
        CONSTRAINT CK_readmrz_face_match_jobs_Thresholds CHECK (
            ReviewThreshold >= -1.0 AND ReviewThreshold <= 1.0
            AND MatchThreshold >= -1.0 AND MatchThreshold <= 1.0
            AND MatchThreshold >= ReviewThreshold
        ),
        CONSTRAINT CK_readmrz_face_match_jobs_MetadataJson
            CHECK (MetadataJson IS NULL OR ISJSON(MetadataJson) = 1)
    );
END;
GO

IF OBJECT_ID(N'dbo.readmrz_face_match_items', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_face_match_items (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_face_match_items PRIMARY KEY,
        ItemKey UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT DF_readmrz_face_match_items_ItemKey DEFAULT NEWSEQUENTIALID(),
        JobId BIGINT NOT NULL,
        SequenceNo INT NOT NULL,
        OriginalFileName NVARCHAR(512) NOT NULL,
        RelativeImagePath NVARCHAR(1000) NOT NULL,
        RelativeEmbeddingPath NVARCHAR(1000) NULL,
        ContentType NVARCHAR(128) NULL,
        FileSizeBytes BIGINT NULL,
        Sha256 NVARCHAR(64) NULL,
        ImageWidth INT NULL,
        ImageHeight INT NULL,

        Status NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_face_match_items_Status DEFAULT (N'queued'),
        AttemptCount INT NOT NULL CONSTRAINT DF_readmrz_face_match_items_Attempts DEFAULT (0),
        MaxAttempts INT NOT NULL CONSTRAINT DF_readmrz_face_match_items_MaxAttempts DEFAULT (3),
        NextAttemptAt DATETIME2(3) NULL,
        LockedBy NVARCHAR(128) NULL,
        LockedAt DATETIME2(3) NULL,

        ClassificationLabel NVARCHAR(32) NULL,
        ClassificationConfidence FLOAT NULL,
        FaceCount INT NOT NULL CONSTRAINT DF_readmrz_face_match_items_FaceCount DEFAULT (0),
        EmbeddingCount INT NOT NULL CONSTRAINT DF_readmrz_face_match_items_EmbeddingCount DEFAULT (0),
        EmbeddingDimension INT NULL,
        PrimaryFaceConfidence FLOAT NULL,
        ResultJson NVARCHAR(MAX) NULL,
        ErrorCode NVARCHAR(100) NULL,
        ErrorMessage NVARCHAR(2000) NULL,
        ProcessTimeMs INT NULL,

        ProcessStartedAt DATETIME2(3) NULL,
        ProcessedAt DATETIME2(3) NULL,
        CreatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_face_match_items_Created DEFAULT SYSUTCDATETIME(),
        UpdatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_face_match_items_Updated DEFAULT SYSUTCDATETIME(),
        RowVersion ROWVERSION NOT NULL,

        CONSTRAINT FK_readmrz_face_match_items_Job
            FOREIGN KEY (JobId) REFERENCES dbo.readmrz_face_match_jobs(Id),
        CONSTRAINT UQ_readmrz_face_match_items_ItemKey UNIQUE (ItemKey),
        CONSTRAINT UQ_readmrz_face_match_items_JobSequence UNIQUE (JobId, SequenceNo),
        CONSTRAINT CK_readmrz_face_match_items_Status CHECK (Status IN (
            N'queued', N'processing', N'retry_wait', N'succeeded', N'failed', N'cancelled'
        )),
        CONSTRAINT CK_readmrz_face_match_items_Class CHECK (
            ClassificationLabel IS NULL OR ClassificationLabel IN (N'face', N'passport', N'unknown')
        ),
        CONSTRAINT CK_readmrz_face_match_items_ResultJson
            CHECK (ResultJson IS NULL OR ISJSON(ResultJson) = 1)
    );
END;
GO

IF OBJECT_ID(N'dbo.readmrz_face_match_pairs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_face_match_pairs (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_face_match_pairs PRIMARY KEY,
        PairKey UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT DF_readmrz_face_match_pairs_PairKey DEFAULT NEWSEQUENTIALID(),
        JobId BIGINT NOT NULL,
        FaceItemId BIGINT NULL,
        PassportItemId BIGINT NULL,
        FaceDetectionIndex INT NULL,
        PassportDetectionIndex INT NULL,
        Score FLOAT NULL,
        Decision NVARCHAR(32) NOT NULL,
        ReviewRequired BIT NOT NULL CONSTRAINT DF_readmrz_face_match_pairs_Review DEFAULT (0),
        ResultJson NVARCHAR(MAX) NULL,
        CreatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_face_match_pairs_Created DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_readmrz_face_match_pairs_Job
            FOREIGN KEY (JobId) REFERENCES dbo.readmrz_face_match_jobs(Id),
        CONSTRAINT FK_readmrz_face_match_pairs_FaceItem
            FOREIGN KEY (FaceItemId) REFERENCES dbo.readmrz_face_match_items(Id),
        CONSTRAINT FK_readmrz_face_match_pairs_PassportItem
            FOREIGN KEY (PassportItemId) REFERENCES dbo.readmrz_face_match_items(Id),
        CONSTRAINT UQ_readmrz_face_match_pairs_PairKey UNIQUE (PairKey),
        CONSTRAINT CK_readmrz_face_match_pairs_Decision CHECK (Decision IN (
            N'match', N'review', N'unmatched_face', N'unmatched_passport'
        )),
        CONSTRAINT CK_readmrz_face_match_pairs_ResultJson
            CHECK (ResultJson IS NULL OR ISJSON(ResultJson) = 1),
        CONSTRAINT CK_readmrz_face_match_pairs_Items CHECK (
            (Decision IN (N'match', N'review') AND FaceItemId IS NOT NULL AND PassportItemId IS NOT NULL)
            OR (Decision = N'unmatched_face' AND FaceItemId IS NOT NULL AND PassportItemId IS NULL)
            OR (Decision = N'unmatched_passport' AND FaceItemId IS NULL AND PassportItemId IS NOT NULL)
        )
    );
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_readmrz_face_match_jobs_ClientRequestId'
      AND object_id = OBJECT_ID(N'dbo.readmrz_face_match_jobs')
)
BEGIN
    CREATE UNIQUE INDEX UX_readmrz_face_match_jobs_ClientRequestId
    ON dbo.readmrz_face_match_jobs (ClientRequestId)
    WHERE ClientRequestId IS NOT NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_face_match_items_WorkQueue'
      AND object_id = OBJECT_ID(N'dbo.readmrz_face_match_items')
)
BEGIN
    CREATE INDEX IX_readmrz_face_match_items_WorkQueue
    ON dbo.readmrz_face_match_items (Status, NextAttemptAt, Id)
    INCLUDE (JobId, RelativeImagePath, AttemptCount, MaxAttempts);
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_face_match_jobs_Status'
      AND object_id = OBJECT_ID(N'dbo.readmrz_face_match_jobs')
)
BEGIN
    CREATE INDEX IX_readmrz_face_match_jobs_Status
    ON dbo.readmrz_face_match_jobs (Status, CreatedDate, Id)
    INCLUDE (JobKey, UploadCompleted, TotalItems, SucceededItems, FailedItems);
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_face_match_pairs_Job'
      AND object_id = OBJECT_ID(N'dbo.readmrz_face_match_pairs')
)
BEGIN
    CREATE INDEX IX_readmrz_face_match_pairs_Job
    ON dbo.readmrz_face_match_pairs (JobId, Decision, Score DESC)
    INCLUDE (PairKey, FaceItemId, PassportItemId, ReviewRequired);
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_readmrz_face_match_pairs_Face'
      AND object_id = OBJECT_ID(N'dbo.readmrz_face_match_pairs')
)
BEGIN
    CREATE UNIQUE INDEX UX_readmrz_face_match_pairs_Face
    ON dbo.readmrz_face_match_pairs (JobId, FaceItemId)
    WHERE FaceItemId IS NOT NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'UX_readmrz_face_match_pairs_Passport'
      AND object_id = OBJECT_ID(N'dbo.readmrz_face_match_pairs')
)
BEGIN
    CREATE UNIQUE INDEX UX_readmrz_face_match_pairs_Passport
    ON dbo.readmrz_face_match_pairs (JobId, PassportItemId)
    WHERE PassportItemId IS NOT NULL;
END;
GO
