/*
    Evaluation tables for passport images exported by QuickPassportImageExportService.
    Source images remain on disk. SQL Server stores the run manifest, OCR result and
    comparison between the source folder label and the pipeline classification.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

IF OBJECT_ID(N'dbo.readmrz_passport_evaluation_runs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_passport_evaluation_runs (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_passport_evaluation_runs PRIMARY KEY,
        RunKey UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_runs_RunKey DEFAULT NEWSEQUENTIALID(),
        RunName NVARCHAR(255) NULL,
        SourceRoot NVARCHAR(2048) NOT NULL,
        ReferenceDate DATE NOT NULL,
        Status NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_runs_Status DEFAULT (N'created'),
        TotalItems INT NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_runs_TotalItems DEFAULT (0),
        ConfigJson NVARCHAR(MAX) NULL,
        ErrorMessage NVARCHAR(4000) NULL,
        StartedAt DATETIME2(3) NULL,
        CompletedAt DATETIME2(3) NULL,
        CreatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_runs_CreatedDate DEFAULT SYSUTCDATETIME(),
        UpdatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_runs_UpdatedDate DEFAULT SYSUTCDATETIME(),

        CONSTRAINT UQ_readmrz_passport_evaluation_runs_RunKey UNIQUE (RunKey),
        CONSTRAINT CK_readmrz_passport_evaluation_runs_Status CHECK (
            Status IN (N'created', N'running', N'completed', N'completed_with_errors', N'failed', N'cancelled')
        ),
        CONSTRAINT CK_readmrz_passport_evaluation_runs_TotalItems CHECK (TotalItems >= 0),
        CONSTRAINT CK_readmrz_passport_evaluation_runs_ConfigJson CHECK (
            ConfigJson IS NULL OR ISJSON(ConfigJson) = 1
        )
    );
END;
GO

IF OBJECT_ID(N'dbo.readmrz_passport_evaluation_items', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_passport_evaluation_items (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_passport_evaluation_items PRIMARY KEY,
        ItemKey UNIQUEIDENTIFIER NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_items_ItemKey DEFAULT NEWSEQUENTIALID(),
        RunId BIGINT NOT NULL,
        SequenceNo INT NOT NULL,
        CountryCode NVARCHAR(20) NOT NULL,
        SourceGroup NVARCHAR(64) NOT NULL,
        ExpectedValidityStatus NVARCHAR(40) NOT NULL,
        RelativeImagePath NVARCHAR(2048) NOT NULL,
        RelativePathHash CHAR(64) NOT NULL,
        OriginalFileName NVARCHAR(512) NOT NULL,
        FileSizeBytes BIGINT NOT NULL,
        FileSha256 CHAR(64) NULL,
        Status NVARCHAR(20) NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_items_Status DEFAULT (N'queued'),
        ReadSuccess BIT NULL,
        PipelineFound BIT NULL,
        ChecksumValid BIT NULL,
        IsValidityMatch BIT NULL,
        IsCountryMatch BIT NULL,
        ImageWidth INT NULL,
        ImageHeight INT NULL,
        OrientationDegree INT NULL,
        YoloConfidence FLOAT NULL,
        OcrConfidence FLOAT NULL,
        UsedFallback BIT NULL,
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
        AgeYears INT NULL,
        AgeGroup NVARCHAR(20) NULL,
        ActualValidityStatus NVARCHAR(40) NULL,
        DaysToExpiry INT NULL,
        ResultJson NVARCHAR(MAX) NULL,
        ErrorCode NVARCHAR(100) NULL,
        ErrorMessage NVARCHAR(4000) NULL,
        ProcessedAt DATETIME2(3) NULL,
        CreatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_items_CreatedDate DEFAULT SYSUTCDATETIME(),
        UpdatedDate DATETIME2(3) NOT NULL
            CONSTRAINT DF_readmrz_passport_evaluation_items_UpdatedDate DEFAULT SYSUTCDATETIME(),

        CONSTRAINT FK_readmrz_passport_evaluation_items_Run
            FOREIGN KEY (RunId) REFERENCES dbo.readmrz_passport_evaluation_runs(Id),
        CONSTRAINT UQ_readmrz_passport_evaluation_items_ItemKey UNIQUE (ItemKey),
        CONSTRAINT UQ_readmrz_passport_evaluation_items_RunSequence UNIQUE (RunId, SequenceNo),
        CONSTRAINT UQ_readmrz_passport_evaluation_items_RunPath UNIQUE (RunId, RelativePathHash),
        CONSTRAINT CK_readmrz_passport_evaluation_items_Status CHECK (
            Status IN (N'queued', N'processing', N'succeeded', N'failed', N'cancelled')
        ),
        CONSTRAINT CK_readmrz_passport_evaluation_items_ExpectedValidity CHECK (
            ExpectedValidityStatus IN (N'expired', N'expiring_within_6_months', N'valid_6_months_or_more')
        ),
        CONSTRAINT CK_readmrz_passport_evaluation_items_ActualValidity CHECK (
            ActualValidityStatus IS NULL OR ActualValidityStatus IN (
                N'expired', N'expiring_within_6_months', N'valid_6_months_or_more', N'unknown'
            )
        ),
        CONSTRAINT CK_readmrz_passport_evaluation_items_AgeGroup CHECK (
            AgeGroup IS NULL OR AgeGroup IN (N'child', N'adult', N'unknown')
        ),
        CONSTRAINT CK_readmrz_passport_evaluation_items_FileSize CHECK (FileSizeBytes >= 0),
        CONSTRAINT CK_readmrz_passport_evaluation_items_ResultJson CHECK (
            ResultJson IS NULL OR ISJSON(ResultJson) = 1
        )
    );
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_evaluation_items_RunStatus'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_evaluation_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_evaluation_items_RunStatus
        ON dbo.readmrz_passport_evaluation_items (RunId, Status, SequenceNo);
END;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_readmrz_passport_evaluation_items_Mismatch'
      AND object_id = OBJECT_ID(N'dbo.readmrz_passport_evaluation_items')
)
BEGIN
    CREATE INDEX IX_readmrz_passport_evaluation_items_Mismatch
        ON dbo.readmrz_passport_evaluation_items (RunId, IsValidityMatch, CountryCode)
        INCLUDE (RelativeImagePath, ExpectedValidityStatus, ActualValidityStatus, ErrorMessage);
END;
GO

IF OBJECT_ID(N'dbo.vw_readmrz_passport_evaluation_summary', N'V') IS NULL
BEGIN
    EXEC(N'CREATE VIEW dbo.vw_readmrz_passport_evaluation_summary AS SELECT 1 AS Placeholder;');
END;
GO

ALTER VIEW dbo.vw_readmrz_passport_evaluation_summary
AS
SELECT
    run.Id AS RunId,
    run.RunKey,
    run.RunName,
    run.SourceRoot,
    run.ReferenceDate,
    run.Status,
    run.TotalItems,
    COUNT(item.Id) AS ManifestItems,
    SUM(CASE WHEN item.Status IN (N'succeeded', N'failed', N'cancelled') THEN 1 ELSE 0 END) AS ProcessedItems,
    SUM(CASE WHEN item.ReadSuccess = 1 THEN 1 ELSE 0 END) AS ReadSuccessItems,
    SUM(CASE WHEN item.ReadSuccess = 0 OR item.Status = N'failed' THEN 1 ELSE 0 END) AS ReadFailedItems,
    SUM(CASE WHEN item.Status = N'failed' THEN 1 ELSE 0 END) AS SystemErrorItems,
    SUM(CASE WHEN item.ChecksumValid = 1 THEN 1 ELSE 0 END) AS ChecksumValidItems,
    SUM(CASE WHEN item.IsValidityMatch = 1 THEN 1 ELSE 0 END) AS ValidityCorrectItems,
    SUM(CASE WHEN item.IsValidityMatch = 0 THEN 1 ELSE 0 END) AS ValidityIncorrectItems,
    CAST(
        CASE WHEN SUM(CASE WHEN item.IsValidityMatch IS NOT NULL THEN 1 ELSE 0 END) = 0 THEN 0
             ELSE 100.0 * SUM(CASE WHEN item.IsValidityMatch = 1 THEN 1 ELSE 0 END)
                  / SUM(CASE WHEN item.IsValidityMatch IS NOT NULL THEN 1 ELSE 0 END)
        END AS DECIMAL(6,2)
    ) AS ValidityAccuracyPercent,
    SUM(CASE WHEN item.IsCountryMatch = 1 THEN 1 ELSE 0 END) AS CountryCorrectItems,
    SUM(CASE WHEN item.IsCountryMatch = 0 THEN 1 ELSE 0 END) AS CountryIncorrectItems,
    SUM(CASE WHEN item.AgeGroup = N'child' THEN 1 ELSE 0 END) AS ChildPassportItems,
    SUM(CASE WHEN item.AgeGroup = N'adult' THEN 1 ELSE 0 END) AS AdultPassportItems,
    SUM(CASE WHEN item.AgeGroup = N'unknown' THEN 1 ELSE 0 END) AS UnknownAgeItems,
    SUM(CASE WHEN item.ActualValidityStatus = N'expired' THEN 1 ELSE 0 END) AS ExpiredPassportItems,
    SUM(CASE WHEN item.ActualValidityStatus = N'expiring_within_6_months' THEN 1 ELSE 0 END) AS UnderSixMonthsItems,
    SUM(CASE WHEN item.ActualValidityStatus = N'valid_6_months_or_more' THEN 1 ELSE 0 END) AS SixMonthsOrMoreItems,
    SUM(CASE WHEN item.ActualValidityStatus = N'unknown' THEN 1 ELSE 0 END) AS UnknownValidityItems,
    run.StartedAt,
    run.CompletedAt,
    run.CreatedDate,
    run.UpdatedDate
FROM dbo.readmrz_passport_evaluation_runs run
LEFT JOIN dbo.readmrz_passport_evaluation_items item ON item.RunId = run.Id
GROUP BY
    run.Id, run.RunKey, run.RunName, run.SourceRoot, run.ReferenceDate, run.Status,
    run.TotalItems, run.StartedAt, run.CompletedAt, run.CreatedDate, run.UpdatedDate;
GO

IF OBJECT_ID(N'dbo.vw_readmrz_passport_evaluation_issues', N'V') IS NULL
BEGIN
    EXEC(N'CREATE VIEW dbo.vw_readmrz_passport_evaluation_issues AS SELECT 1 AS Placeholder;');
END;
GO

ALTER VIEW dbo.vw_readmrz_passport_evaluation_issues
AS
SELECT
    run.Id AS RunId,
    run.RunKey,
    run.RunName,
    run.ReferenceDate,
    item.Id AS ItemId,
    item.ItemKey,
    item.SequenceNo,
    item.CountryCode,
    item.SourceGroup,
    item.RelativeImagePath,
    item.Status,
    item.ReadSuccess,
    item.ChecksumValid,
    item.ExpectedValidityStatus,
    item.ActualValidityStatus,
    item.IsValidityMatch,
    item.Nationality,
    item.IssuingCountry,
    item.IsCountryMatch,
    item.AgeGroup,
    item.DateOfBirth,
    item.DateOfExpiry,
    item.ErrorCode,
    item.ErrorMessage,
    item.ProcessTimeMs,
    item.ProcessedAt
FROM dbo.readmrz_passport_evaluation_runs run
INNER JOIN dbo.readmrz_passport_evaluation_items item ON item.RunId = run.Id
WHERE item.Status = N'failed'
   OR item.ReadSuccess = 0
   OR item.IsValidityMatch = 0
   OR item.IsCountryMatch = 0;
GO

SELECT N'Passport evaluation schema is ready.' AS Result;
