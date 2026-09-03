IF OBJECT_ID(N'dbo.readmrz_vn_visa_items', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_vn_visa_items (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_vn_visa_items PRIMARY KEY,

        SourceTable NVARCHAR(128) NULL,
        SourceId BIGINT NULL,
        TransactionEVisaId BIGINT NULL,
        TransactionGuid UNIQUEIDENTIFIER NULL,
        SourceKey NVARCHAR(700) NOT NULL,

        DocumentType NVARCHAR(32) NOT NULL,
        RelativeImagePath NVARCHAR(1000) NOT NULL,
        OrientedRelativeImagePath NVARCHAR(1000) NULL,
        OriginalFileName NVARCHAR(512) NULL,
        ImageWidth INT NULL,
        ImageHeight INT NULL,
        FileSizeBytes BIGINT NULL,
        Sha256 NVARCHAR(64) NULL,

        Split NVARCHAR(16) NULL,
        Status NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_vn_visa_items_Status DEFAULT (N'pending'),
        ReviewStatus NVARCHAR(32) NOT NULL
            CONSTRAINT DF_readmrz_vn_visa_items_ReviewStatus DEFAULT (N'pending'),

        OcrEngine NVARCHAR(128) NULL,
        OcrConfigJson NVARCHAR(MAX) NULL,
        OrientationJson NVARCHAR(MAX) NULL,
        RawOcrJson NVARCHAR(MAX) NULL,

        RegexVersion NVARCHAR(64) NULL,
        AutoMappingVersion NVARCHAR(64) NULL,
        DataJson NVARCHAR(MAX) NULL,
        FinalDataJson NVARCHAR(MAX) NULL,

        ProcessStartedAt DATETIME2(0) NULL,
        ProcessedAt DATETIME2(0) NULL,
        ReviewedAt DATETIME2(0) NULL,
        ReviewedBy NVARCHAR(128) NULL,
        ErrorMessage NVARCHAR(2000) NULL,

        CreatedDate DATETIME2(0) NOT NULL
            CONSTRAINT DF_readmrz_vn_visa_items_CreatedDate DEFAULT SYSDATETIME(),
        UpdatedDate DATETIME2(0) NOT NULL
            CONSTRAINT DF_readmrz_vn_visa_items_UpdatedDate DEFAULT SYSDATETIME(),

        CONSTRAINT UQ_readmrz_vn_visa_items_SourceKey UNIQUE (SourceKey),

        CONSTRAINT CK_readmrz_vn_visa_items_DocumentType
            CHECK (DocumentType IN (N'visa_sticker_vn', N'loose_visa_vn')),

        CONSTRAINT CK_readmrz_vn_visa_items_Status
            CHECK (Status IN (N'pending', N'ocr_done', N'mapped', N'exported', N'skipped', N'error')),

        CONSTRAINT CK_readmrz_vn_visa_items_ReviewStatus
            CHECK (ReviewStatus IN (N'pending', N'needs_review', N'approved', N'rejected')),

        CONSTRAINT CK_readmrz_vn_visa_items_Split
            CHECK (Split IS NULL OR Split IN (N'train', N'val', N'test')),

        CONSTRAINT CK_readmrz_vn_visa_items_RelativeImagePath
            CHECK (
                RelativeImagePath NOT LIKE N'[A-Za-z]:%'
                AND LEFT(RelativeImagePath, 1) NOT IN (N'/', N'\')
                AND RelativeImagePath NOT LIKE N'\\%'
            ),

        CONSTRAINT CK_readmrz_vn_visa_items_OrientedRelativeImagePath
            CHECK (
                OrientedRelativeImagePath IS NULL
                OR (
                    OrientedRelativeImagePath NOT LIKE N'[A-Za-z]:%'
                    AND LEFT(OrientedRelativeImagePath, 1) NOT IN (N'/', N'\')
                    AND OrientedRelativeImagePath NOT LIKE N'\\%'
                )
            ),

        CONSTRAINT CK_readmrz_vn_visa_items_OcrConfigJson
            CHECK (OcrConfigJson IS NULL OR ISJSON(OcrConfigJson) = 1),

        CONSTRAINT CK_readmrz_vn_visa_items_OrientationJson
            CHECK (OrientationJson IS NULL OR ISJSON(OrientationJson) = 1),

        CONSTRAINT CK_readmrz_vn_visa_items_RawOcrJson
            CHECK (RawOcrJson IS NULL OR ISJSON(RawOcrJson) = 1),

        CONSTRAINT CK_readmrz_vn_visa_items_DataJson
            CHECK (DataJson IS NULL OR ISJSON(DataJson) = 1),

        CONSTRAINT CK_readmrz_vn_visa_items_FinalDataJson
            CHECK (FinalDataJson IS NULL OR ISJSON(FinalDataJson) = 1)
    );
END;
GO

IF COL_LENGTH(N'dbo.readmrz_vn_visa_items', N'OrientedRelativeImagePath') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_vn_visa_items
    ADD OrientedRelativeImagePath NVARCHAR(1000) NULL;
END;
GO

IF COL_LENGTH(N'dbo.readmrz_vn_visa_items', N'OrientationJson') IS NULL
BEGIN
    ALTER TABLE dbo.readmrz_vn_visa_items
    ADD OrientationJson NVARCHAR(MAX) NULL;
END;
GO

IF OBJECT_ID(N'dbo.readmrz_vn_visa_review_history', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.readmrz_vn_visa_review_history (
        Id BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_readmrz_vn_visa_review_history PRIMARY KEY,

        VisaItemId BIGINT NOT NULL,
        Decision NVARCHAR(32) NOT NULL,
        FieldName NVARCHAR(64) NULL,

        OldDataJson NVARCHAR(MAX) NULL,
        NewDataJson NVARCHAR(MAX) NULL,
        Note NVARCHAR(1000) NULL,
        ReviewedBy NVARCHAR(128) NULL,

        CreatedDate DATETIME2(0) NOT NULL
            CONSTRAINT DF_readmrz_vn_visa_review_history_CreatedDate DEFAULT SYSDATETIME(),

        CONSTRAINT FK_readmrz_vn_visa_review_history_Item
            FOREIGN KEY (VisaItemId) REFERENCES dbo.readmrz_vn_visa_items(Id),

        CONSTRAINT CK_readmrz_vn_visa_review_history_Decision
            CHECK (Decision IN (N'approved', N'rejected', N'updated_bbox', N'updated_value', N'needs_review')),

        CONSTRAINT CK_readmrz_vn_visa_review_history_OldDataJson
            CHECK (OldDataJson IS NULL OR ISJSON(OldDataJson) = 1),

        CONSTRAINT CK_readmrz_vn_visa_review_history_NewDataJson
            CHECK (NewDataJson IS NULL OR ISJSON(NewDataJson) = 1)
    );
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_vn_visa_items_ReviewQueue'
      AND object_id = OBJECT_ID(N'dbo.readmrz_vn_visa_items')
)
BEGIN
    CREATE INDEX IX_readmrz_vn_visa_items_ReviewQueue
    ON dbo.readmrz_vn_visa_items (ReviewStatus, Status, Id)
    INCLUDE (DocumentType, RelativeImagePath, Split);
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_vn_visa_items_Sha256'
      AND object_id = OBJECT_ID(N'dbo.readmrz_vn_visa_items')
)
BEGIN
    CREATE INDEX IX_readmrz_vn_visa_items_Sha256
    ON dbo.readmrz_vn_visa_items (Sha256)
    WHERE Sha256 IS NOT NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_vn_visa_items_TransactionEVisaId'
      AND object_id = OBJECT_ID(N'dbo.readmrz_vn_visa_items')
)
BEGIN
    CREATE INDEX IX_readmrz_vn_visa_items_TransactionEVisaId
    ON dbo.readmrz_vn_visa_items (TransactionEVisaId)
    WHERE TransactionEVisaId IS NOT NULL;
END;
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'IX_readmrz_vn_visa_review_history_Item'
      AND object_id = OBJECT_ID(N'dbo.readmrz_vn_visa_review_history')
)
BEGIN
    CREATE INDEX IX_readmrz_vn_visa_review_history_Item
    ON dbo.readmrz_vn_visa_review_history (VisaItemId, CreatedDate DESC);
END;
GO
