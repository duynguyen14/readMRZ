IF OBJECT_ID(N'dbo.readmrz_pipeline_test_items', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_pipeline_test_items (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_pipeline_test_items PRIMARY KEY,

        TransactionEVisaId BIGINT NULL,
        TransactionGuid UNIQUEIDENTIFIER NULL,
        PassportNo NVARCHAR(100) NULL,

        SourceMrzlineOne NVARCHAR(128) NULL,
        SourceMrzlineTwo NVARCHAR(128) NULL,
        SourceMrzlineOnePoint FLOAT NULL,
        SourceMrzlineTwoPoint FLOAT NULL,

        PredictedMrzlineOne NVARCHAR(128) NULL,
        PredictedMrzlineTwo NVARCHAR(128) NULL,
        LineOneConfidence FLOAT NULL,
        LineTwoConfidence FLOAT NULL,

        IsLineOneMatch BIT NOT NULL
            CONSTRAINT DF_readmrz_pipeline_test_items_IsLineOneMatch DEFAULT (0),
        IsLineTwoMatch BIT NOT NULL
            CONSTRAINT DF_readmrz_pipeline_test_items_IsLineTwoMatch DEFAULT (0),
        IsFullMatch BIT NOT NULL
            CONSTRAINT DF_readmrz_pipeline_test_items_IsFullMatch DEFAULT (0),
        ParseChecksumOk BIT NOT NULL
            CONSTRAINT DF_readmrz_pipeline_test_items_ParseChecksumOk DEFAULT (0),

        ParsedPassportType NVARCHAR(20) NULL,
        ParsedPassportNo NVARCHAR(50) NULL,
        ParsedFullName NVARCHAR(255) NULL,
        ParsedDob DATE NULL,
        ParsedGender NVARCHAR(20) NULL,
        ParsedNationality NVARCHAR(20) NULL,
        ParsedExpireDate DATE NULL,
        ParsedIssuerCountry NVARCHAR(20) NULL,

        ImagePath NVARCHAR(1000) NULL,
        RawMrzCropPath NVARCHAR(1000) NULL,
        DeskewedMrzCropPath NVARCHAR(1000) NULL,
        LineOneCropPath NVARCHAR(1000) NULL,
        LineTwoCropPath NVARCHAR(1000) NULL,

        YoloConfidence FLOAT NULL,
        OrientationDegree INT NULL,
        UsedFallback BIT NOT NULL
            CONSTRAINT DF_readmrz_pipeline_test_items_UsedFallback DEFAULT (0),
        FallbackReason NVARCHAR(1000) NULL,
        ProcessTimeMs INT NULL,
        ErrorMessage NVARCHAR(MAX) NULL,

        CreatedDate DATETIME2(0) NOT NULL
            CONSTRAINT DF_readmrz_pipeline_test_items_CreatedDate DEFAULT SYSDATETIME()
    );
END;

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_pipeline_test_items_IsFullMatch'
      AND object_id = OBJECT_ID(N'dbo.readmrz_pipeline_test_items')
)
BEGIN
    CREATE INDEX IX_readmrz_pipeline_test_items_IsFullMatch
    ON dbo.readmrz_pipeline_test_items (IsFullMatch, CreatedDate DESC);
END;

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_pipeline_test_items_TransactionEVisaId'
      AND object_id = OBJECT_ID(N'dbo.readmrz_pipeline_test_items')
)
BEGIN
    CREATE INDEX IX_readmrz_pipeline_test_items_TransactionEVisaId
    ON dbo.readmrz_pipeline_test_items (TransactionEVisaId);
END;

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_pipeline_test_items_CreatedDate'
      AND object_id = OBJECT_ID(N'dbo.readmrz_pipeline_test_items')
)
BEGIN
    CREATE INDEX IX_readmrz_pipeline_test_items_CreatedDate
    ON dbo.readmrz_pipeline_test_items (CreatedDate DESC);
END;
